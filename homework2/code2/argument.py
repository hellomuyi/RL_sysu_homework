def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="PongNoFrameskip-v4", help='environment name')

    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--buffer_size", default=int(1e5), type=int)
    parser.add_argument("--lr", default=3e-4, type=float)   # 0.00025
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=5, type=float)
    parser.add_argument("--eps_begin", default=0.9, type=float)
    parser.add_argument("--eps_end", default=0.01, type=float)
    parser.add_argument("--num_episodes", default=401, type=int)
    parser.add_argument("--update_target", default=500, type=int)
    parser.add_argument("--break_step", default=int(1e5), type=int)


    parser.add_argument("--test", default=True, type=bool)  # 测试时加载已训练好的模型
    # parser.add_argument("--use_cuda", default=True, type=bool)
    # parser.add_argument("--n_frames", default=int(400000), type=int)
    # parser.add_argument("--learning_freq", default=1, type=int)
    # parser.add_argument("--target_update_freq", default=40000, type=int)
    parser.add_argument("--exp_name", default='double_dqn1', type=str)
    parser.add_argument("--log_dir", default='./result/train/', type=str)
    parser.add_argument("--save_dir", default='./result/train/', type=str)
    parser.add_argument("--model_dir", default='./result/checkpoints/', type=str)

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=1, type=float)
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--with_baseline", default=True, type=bool)

    parser.add_argument("--test", default=True, type=bool)
    parser.add_argument("--exp_name", default='pg2_seed1', type=str)
    parser.add_argument("--log_dir", default='./result/train/', type=str)
    parser.add_argument("--save_dir", default='./result/train/', type=str)
    parser.add_argument("--model_dir", default='./result/checkpoints/', type=str)

    return parser
