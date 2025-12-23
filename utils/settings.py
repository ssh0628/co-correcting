import sys
import argparse

def get_args():
    if sys.platform == 'darwin':
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/ISIC-Archive-Downloader/Data_sample_balanced'
        mnist_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/MNIST'
        cifar10_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar10'
        cifar100_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar100'
        pcam_root = "/Users/jiarunliu/Documents/BUCT/Label_517/dataset/PatchCamelyon"
        batch_size = 8
        device = 'cpu'
        data_device = 0
        noise_type = 'sn'
        stage1 = 1
        stage2 = 3

    elif sys.platform == 'linux':
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/home/fgldlb/Documents/ISIC-Archive-Downloader/NewData'
        pcam_root = "/home/fgldlb/Documents/dataset/PatchCamelyon"
        mnist_root = './data/mnist'
        cifar10_root = './data/cifar10'
        cifar100_root = './data/cifar100'
        batch_size = 32
        device = 'cuda:0'
        data_device = 1
        noise_type = 'sn'
        stage1 = 70
        stage2 = 200

    elif sys.platform == 'win32':
        # Windows 전용 기본 설정
        isic_root = 'D:/Datasets/ISIC'  # 예시 경로
        petskin_root = 'D:/Datasets/PetSkin'
        mnist_root = './data/mnist'
        batch_size = 16
        device = 'cuda:0' # GPU 사용 시
        data_device = 0
        noise_type = 'clean'
        stage1 = 70
        stage2 = 200
        
    else:
        # 기본 설정 (Default)
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = None
        mnist_root = './data/mnist'
        cifar10_root = '/data/cifar10'
        cifar100_root = '/data/cifar100'
        pcam_root = None
        batch_size = 16
        device = 'cpu'
        data_device = 0
        noise_type = 'clean'
        stage1 = 70
        stage2 = 200

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # 일반적인 학습 파라미터(Normal parameters)
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                        metavar='N', help='미니 배치 크기 (기본값: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='H-P', help='초기 학습률 (Initial Learning Rate)')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-5, type=float,
                        metavar='H-P', help='Stage 3(파인튜닝)에서의 학습률')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='모멘텀 (Momentum), SGD 등에서 사용')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='가중치 감쇠 (Weight Decay), 과적합 방지 (기본값: 1e-4)')
    parser.add_argument('--backbone', dest="backbone", default="resnet50", type=str,
                        help="사용할 백본 네트워크 모델 (예: resnet50)")
    parser.add_argument('--optim', dest="optim", default="SGD", type=str,
                        choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'Adadelta', 'Adagrad', 'mix'],
                        help="최적화 알고리즘 (Optimizer) 선택")
    parser.add_argument('--scheduler', dest='scheduler', default=None, type=str, choices=['cyclic', None, "SWA"],
                        help="학습률 스케줄러 선택")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='데이터 로딩에 사용할 워커 프로세스 수 (기본값: 4)')
    
    # Co-teaching 파라미터 (라벨 노이즈 처리를 위한 핵심 설정)
    """
        --forget rate : R(t) = (1 - forget rate)
        --num-gradual : E_k
    """
    parser.add_argument('--forget-rate', '--fr', '--forget_rate', default=0.2, type=float,
                        metavar='H-P', help='망각(Forget) 비율. 노이즈 비율과 비슷하게 설정하는 것을 권장.')
    parser.add_argument('--num-gradual', '--ng', '--num_gradual', default=10, type=int,
                        metavar='H-P', help='선형적으로 망각 비율을 높여갈 에폭 수 (Tk).')
    parser.add_argument('--exponent', default=1, type=float,
                        metavar='H-P', help='망각 비율 증가 지수 (1이면 선형).')
    parser.add_argument('--loss-type', dest="loss_type", default="coteaching_plus", type=str,
                        choices=['coteaching_plus', 'coteaching'],
                        help="손실 함수 타입 선택: [coteaching_plus, coteaching]")
    parser.add_argument('--warmup', '--wm', '--warm-up', default=0, type=float,
                        metavar='H-P', help='웜업(Warm up) 에폭 수. 초기에는 모든 데이터를 신뢰.')
    parser.add_argument('--linear-num', '--linear_num', default=256, type=int,
                        metavar='H-P', help='선형 레이어의 노드 수 (일부 모델용).')
    
    # PENCIL 알고리즘 파라미터 (라벨 수정 및 확률적 모델링)
    """
        --alpha : 호환성 손실 가중치 (compatibility loss weight)
        --beta : 엔트로피 손실 가중치 (entropy loss weight)
        --lambda1 : 라벨 수정 비율 (label correction rate)
    """
    parser.add_argument('--alpha', default=0.4, type=float,
                        metavar='H-P', help='호환성 손실(Compatibility Loss)의 계수(alpha)')
    parser.add_argument('--beta', default=0.1, type=float,
                        metavar='H-P', help='엔트로피 손실(Entropy Loss)의 계수(beta)')
    parser.add_argument('--lambda1', default=200, type=int,
                        metavar='H-P', help='라벨 수정(Label Correction) 강도 (lambda), 값이 클수록 원래 라벨을 덜 신뢰')
    parser.add_argument('--K', default=10.0, type=float, )
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='학습을 시작할 에폭 번호 (재시작 시 유용)')
    parser.add_argument('--epochs', default=320, type=int, metavar='H-P',
                        help='총 학습 에폭 수')
    parser.add_argument('--stage1', default=stage1, type=int,
                        metavar='H-P', help='Stage 1 (Warm up 및 초기 학습) 종료 에폭')
    parser.add_argument('--stage2', default=stage2, type=int,
                        metavar='H-P', help='Stage 2 (라벨 수정 단계) 종료 에폭')
    
    # 노이즈(Noise) 설정
    parser.add_argument('--noise', default=0.20, type=float,
                        help='데이터 라벨의 노이즈 비율 (실험용)')
    parser.add_argument('--noise_type', default=noise_type,  choices=['clean', 'sn', 'pairflip'],type=str,
                        help='노이즈 타입 (clean: 노이즈 없음, sn: 대칭 노이즈, pairflip: 페어 플립)')
    
    # 데이터(Data) 설정
    parser.add_argument("--dataset", dest="dataset", default='mnist', type=str,
                        choices=['mnist', 'cifar10', 'cifar100', 'cifar2', 'isic', 'clothing1m', 'pcam', 'petskin'],
                        help="사용할 데이터셋 선택 (petskin 포함)")
    parser.add_argument("--image_size", dest="image_size", default=224, type=int,
                        help="입력 이미지 크기 (예: 224)")
    parser.add_argument('--classnum', default=2, type=int,
                        metavar='H-P', help='데이터셋 클래스 개수')
    parser.add_argument('--device', dest='device', default=device, type=str,
                        help='사용할 GPU/CPU 디바이스 (예: cuda:0)')
    parser.add_argument('--data_device', dest="data_device", default=data_device, type=int,
                        help="데이터를 로드할 위치 (0: 디스크에서 읽기, 1: RAM에 미리 로드). 메모리 부족 시 0 권장.")
    parser.add_argument('--dataRoot',dest='root',default=isic_root,
                        type=str,metavar='PATH',help='데이터셋 위치(경로)')
    parser.add_argument('--datanum', default=15000, type=int,
                        metavar='H-P', help='학습 데이터 샘플 수')
    parser.add_argument('--train-redux', dest="train_redux", default=None, type=int,
                        help='train data number, default None')
    parser.add_argument('--test-redux', dest="test_redux", default=None, type=int,
                        help='test data number, default None')
    parser.add_argument('--val-redux', dest="val_redux", default=None, type=int,
                        help='validate data number, default None')
    parser.add_argument('--full-test', dest="full_test", default=False, type=bool,
                        help='전체 테스트 데이터셋 사용 여부 (기본값: False)')
    parser.add_argument('--random-ind-redux', dest="random_ind_redux", default=False, type=bool,
                        help='테스트/검증 데이터 축소 시 무작위 선택 여부 (기본값: False)')
    # Curriculum settings (커리큘럼 학습 설정)
    parser.add_argument("--curriculum", dest="curriculum", default=1, type=int,
                        help="라벨 업데이트 시 커리큘럼 적용 여부")
    parser.add_argument("--cluster-mode", dest="cluster_mode", default='dual', type=str, choices=['dual', 'single', 'dual_PCA'],
                        help="클러스터링 모드 (dual: 두 모델 특징 결합, single: 단일 모델, dual_PCA: PCA 적용)")
    parser.add_argument("--dim-reduce", dest="dim_reduce", default=256, type=int,
                        help="PCA를 이용한 커리큘럼 특징 차원 축소 크기")
    parser.add_argument("--mix-grad", dest="mix_grad", default=1, type=int,
                        help="두 모델의 그라디언트 혼합 여부 (1=True)")
    parser.add_argument("--discard", dest="discard", default=0, type=int,
                        help="버려진(Discard) 샘플의 라벨만 업데이트할지 여부 (1=True)")
    parser.add_argument("--gamma", dest="gamma", default=0.6, type=int,
                        help="망각 비율(Forget Rate) 스케줄러 파라미터 Gamma")
    parser.add_argument("--finetune-schedule", '-fs', dest="finetune_schedule", default=0, type=int,
                        help="Forget Rate 스케줄러 파인튜닝 적용 여부")
    # trainer settings
    parser.add_argument('--dir', dest='dir', default="experiment/test-debug", type=str,
                        metavar='PATH', help='실험 결과 저장 경로')
    parser.add_argument('--random-seed', dest='random_seed', default=None, type=int,
                        metavar='N', help='PyTorch 랜덤 시드 (기본값: None)')
    args = parser.parse_args()

    # Setting for different dataset
    if args.dataset == "isic":
        print("Training on ISIC")
        args.backbone = 'resnet50'
        args.image_size = 224
        args.classnum = 2
        args.input_dim = 3

    elif args.dataset == 'mnist':
        print("Training on mnist")
        args.backbone = 'cnn'
        if args.root == isic_root:
            args.root = mnist_root
        args.batch_size = 128
        args.image_size = 28
        args.classnum = 10
        args.input_dim = 1
        args.linear_num = 144
        args.datanum = 60000
        args.lr = 0.001
        args.lr2 = 0.0001

    elif args.dataset == 'pcam':
        if args.root == isic_root:
            args.root = pcam_root
        args.backbone = 'densenet169'
        args.batch_size = 128
        args.image_size = 96
        args.dim_reduce = 128
        args.classnum = 2
        args.input_dim = 3
        args.stage1 = 70
        args.stage2 = 200
        args.epochs = 320
        args.datanum = 262144
        args.train_redux = 26214
        args.test_redux = 3276
        args.val_redux = 3276
        args.val_redux = 3276
        args.random_ind_redux = False

    elif args.dataset == 'petskin':
        print("Training on PetSkin")
        if args.root == isic_root and isic_root is not None:
             args.root = petskin_root
        elif args.root is None: # 윈도우 등에서 기본값이 None인 경우
             args.root = petskin_root
             
        args.backbone = 'resnet50' # 기본 모델: ResNet50
        args.image_size = 224 # 이미지 크기
        args.classnum = 6 # 클래스 개수 (A1~A6)
        args.input_dim = 3 # 입력 채널 (RGB)
        # 기본 데이터 개수 설정 (실제 데이터셋 크기에 따라 다를 수 있음)
        args.datanum = 32000



    else:
        print("Use default setting")

    return args