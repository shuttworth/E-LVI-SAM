#include "marginalization_factor.h"

void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        // dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    // std::vector<int> tmp_idx(block_sizes.size());
    // Eigen::MatrixXd tmp(dim, dim);
    // for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //     int size_i = localSize(block_sizes[i]);
    //     Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //     for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //     {
    //         int size_j = localSize(block_sizes[j]);
    //         Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //         tmp_idx[j] = sub_idx;
    //         tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //     }
    // }
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    // std::cout << saes.eigenvalues() << std::endl;
    // ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        // printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    // ROS_WARN("release marginlizationinfo");

    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;

        delete factors[i]->cost_function;

        delete factors[i];
    }
}

/*
addResidualBlockInfo() 这是添加残差信息块的函数
*/
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    // 将该观测项保存；将残差块信息保存到数组
    factors.emplace_back(residual_block_info);

    // 此参数块为传入ResidualBlockInfo的各个参数块地址,提取残差块信息中的所有优化变量地址
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    // 获取代价函数中参与计算的各个参数块的大小(其值为代价函数构造函数的入口参数)；有了参数块的地址和大小，就可以获取各个参数；
    // 通过代价函数获取每个优化变量的size
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    // 对于该残差相关的每一个参数块；residual_block_info->parameter_blocks.size() 为参数块的个数；
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];                        // 获取该参数块的首地址；
        int size = parameter_block_sizes[i];                       // 获取该参数块的大小；
        parameter_block_size[reinterpret_cast<long>(addr)] = size; // 导入优化变量参数<地址，大小>的容器parameter_block_size
    }

    // 对于每一个要边缘化的参数块；drop_set 这个向量中存的是要边缘化的参数块的索引，比如第0块和第1块；
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]]; // 获取要边缘化的参数块的首地址
        // 要边缘化的参数块相对于首地址的ID，即第几个；
        // <地址，索引>容器会首先存放边缘化的优化变量，并且索引值默认为0
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

void MarginalizationInfo::preMarginalize()
{
    for (auto it : factors)
    {
        it->Evaluate();

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

void *ThreadsConstructA(void *threadsstruct)
{
    ThreadsStruct *p = ((ThreadsStruct *)threadsstruct);
    for (auto it : p->sub_factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

// 边缘化函数
void MarginalizationInfo::marginalize()
{
    int pos = 0;
    // 遍历<内存，索引>容器，注意此时<内存，索引>容器内只有边缘化的优化变量
    for (auto &it : parameter_block_idx)
    {
        // 此时给<内存，索引>容器改变索引值，保证每个优化变量地址对应它在容器中的位置
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }

    // 所以m就是边缘化的优化变量的自由度之和
    m = pos;

    // 遍历<内存，大小>容器，此时<内存，大小>容器包含所有的优化变量
    for (const auto &it : parameter_block_size)
    {
        // 如果<内存，索引>容器中不包含这个优化变量，则添加，并且给到正确的索引值
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    // 如果<内存，索引>容器中不包含这个优化变量，则添加，并且给到正确的索引值
    n = pos - m;

    // ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    TicToc t_summing;
    // 构建增量方程Hx=b，这里是Ax=b
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    // multi thread

    // 多线程求解H矩阵
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create(&tids[i], NULL, ThreadsConstructA, (void *)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    // 把多个线程求解的H矩阵加到一起
    for (int i = NUM_THREADS - 1; i >= 0; i--)
    {
        pthread_join(tids[i], NULL);
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }

    // Amm边缘化变量部分对应的矩阵块
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr; // 边缘化之后的系数矩阵（schur补）
    b = brr - Arm * Amm_inv * bmm; // 边缘化之后的常数项

    // 边缘化之后，把Schur构造成J'Jδx=J'e的的形式
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    // Jacobians和residuals
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

// getParameterBlocks()获取保留优化变量的函数
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    // 创建保留优化变量地址数组的容器
    std::vector<double *> keep_block_addr;
    // 清空保留优化变量的其他数据结构
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {
            // 注意到<内存，索引>容器中，索引值大于等于m的，都是边缘化相关的其他优化变量
            // 也就是边缘化操作之后，保留下来的优化变量
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
    // 返回保留优化变量的地址
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info) : marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    // printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    // for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //     //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //     //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    // printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    // printf("residual %x\n", reinterpret_cast<long>(residuals));
    // }
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
