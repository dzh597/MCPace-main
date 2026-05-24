import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K')
    arg.add_argument('-batch_size', type=int, default=1024)
    arg.add_argument('-margin', type=float, default=6.0)
    arg.add_argument('-dim', type=int, default=128)
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str)
    arg.add_argument('-img_dim', type=int, default=4096)
    arg.add_argument('-neg_num', type=int, default=1)
    arg.add_argument('-learning_rate', type=float, default=0.001)
    arg.add_argument('-lrg', type=float, default=0.001)
    arg.add_argument('-lrd', type=float, default=0.001)
    arg.add_argument('-adv_temp', type=float, default=2.0)
    arg.add_argument('-visual', type=str, default='random')
    arg.add_argument('-seed', type=int, default=42)
    arg.add_argument('-missing_rate', type=float, default=0.8)
    arg.add_argument('-postfix', type=str, default='')
    arg.add_argument('-con_temp', type=float, default=0)
    arg.add_argument('-lamda', type=float, default=0)
    arg.add_argument('-mu', type=float, default=0)
    arg.add_argument('-adv_num', type=int, default=1)
    arg.add_argument('-disen_weight', type=float, default=0.01)
    arg.add_argument('-miss_type', type=str, default=None)
    arg.add_argument('-miss_prop', type=float, default=None)
    arg.add_argument('-use_mcpace', type=int, default=0)
    arg.add_argument('-mcpace_mu', type=float, default=0.1)
    arg.add_argument('-mcpace_blocks', type=int, default=4)
    arg.add_argument('-mcpace_lambda_alpha', type=float, default=1.0)
    arg.add_argument('-mcpace_eps', type=float, default=1e-8)
    arg.add_argument('-mcpace_min_rebalance', type=float, default=0.2)
    arg.add_argument('-mcpace_max_rebalance', type=float, default=5.0)
    arg.add_argument('-mcpace_log_interval', type=int, default=0)

    # General gradient-coordination baselines for MCPace comparison.
    # Use -grad_method in {pcgrad,cagrad,nashmtl,alignedmtl,fairgrad}.
    arg.add_argument('-grad_method', type=str, default='none')
    arg.add_argument('-grad_eps', type=float, default=1e-8)
    arg.add_argument('-grad_log_interval', type=int, default=0)

    # CAGrad hyperparameters.
    arg.add_argument('-cagrad_c', type=float, default=0.4)
    arg.add_argument('-cagrad_iters', type=int, default=50)
    arg.add_argument('-cagrad_lr', type=float, default=0.1)

    # Nash-MTL hyperparameters.
    arg.add_argument('-nash_iters', type=int, default=100)
    arg.add_argument('-nash_lr', type=float, default=0.05)
    arg.add_argument('-nash_normalize', type=int, default=1)

    # Aligned-MTL hyperparameters.
    arg.add_argument('-aligned_normalize', type=int, default=1)
    arg.add_argument('-aligned_clamp', type=int, default=1)

    # FairGrad hyperparameters.
    arg.add_argument('-fair_alpha', type=float, default=2.0)
    arg.add_argument('-fair_iters', type=int, default=100)
    arg.add_argument('-fair_lr', type=float, default=0.05)
    arg.add_argument('-fair_normalize', type=int, default=1)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
