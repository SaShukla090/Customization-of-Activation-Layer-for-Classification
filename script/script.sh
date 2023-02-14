cd ..
ls -l
ls
pip install -r requirements.txt

echo "######################################## CONDA ACTIVATED ############################################"
echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = ReLu ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/relu/" --activation_type relurun1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/relu/" --activation_type relurun2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/relu/" --activation_type relurun3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/tanh/" --activation_type tanhrun1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/tanh/" --activation_type tanhrun2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/tanh/" --activation_type tanhrun3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/sigmoid/" --activation_type sigmoidrun1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/sigmoid/" --activation_type sigmoidrun2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/sigmoid/" --activation_type sigmoidrun3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = Adaptive ReLU ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/arelu/" --activation_type arelurun1 
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/arelu/" --activation_type arelurun2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/arelu/" --activation_type arelurun3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = Adaptive Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/atanh/" --activation_type atanhrun1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/atanh/" --activation_type atanhrun2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/atanh/" --activation_type atanhrun3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = Adaptive sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/asigmoid/" --activation_type asigmoidrun1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/asigmoid/" --activation_type asigmoidrun2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/asigmoid/" --activation_type asigmoidrun3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/m1/" --activation_type m1run1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/m1/" --activation_type m1run2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/m1/" --activation_type m1run3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/m2/" --activation_type m2run1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/m2/" --activation_type m2run2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/m2/" --activation_type m2run3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = one with method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onem2/" --activation_type onem2run1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onem2/" --activation_type onem2run2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onem2/" --activation_type onem2run3


echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = one with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onem1/" --activation_type onem1run1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onem1/" --activation_type onem1run2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onem1/" --activation_type onem1run3

echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onethreem1/" --activation_type onethreem1run1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onethreem1/" --activation_type onethreem1run2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onethreem1/" --activation_type onethreem1run3


echo "####################### TRAINING FOR ALEXNET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onethreem2/" --activation_type onethreem2run1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onethreem2/" --activation_type onethreem2run2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onethreem2/" --activation_type onethreem2run3



echo "result for AlexNet"
python sresult_AlexNet.py

echo "############################################### LENET ##############################################################"

echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = ReLu ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/relu/" --activation_type relurun1


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/tanh/" --activation_type tanhrun1


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/sigmoid/" --activation_type sigmoidrun1


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = Adaptive ReLU ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/arelu/" --activation_type arelurun1 


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = Adaptive Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/atanh/" --activation_type atanhrun1

echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = Adaptive sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/asigmoid/" --activation_type asigmoidrun1


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/m1/" --activation_type m1run1

echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/m2/" --activation_type m2run1


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = one with method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/onem2/" --activation_type onem2run1



echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = one with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/AlexNet/onem1/" --activation_type onem1run1


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type AlexNet --model_dir "saved/LeNet/onethreem1/" --activation_type onethreem1run1


echo "####################### TRAINING FOR LENET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type LeNet --model_dir "saved/LeNet/onethreem2/" --activation_type onethreem2run1




echo "#################################################### GOOGLENET ##############################################################"

echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = ReLu ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/relu/" --activation_type relurun1


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/tanh/" --activation_type tanhrun1


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/sigmoid/" --activation_type sigmoidrun1


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = Adaptive ReLU ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/arelu/" --activation_type arelurun1 


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = Adaptive Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/atanh/" --activation_type atanhrun1

echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = Adaptive sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/asigmoid/" --activation_type asigmoidrun1


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/m1/" --activation_type m1run1

echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/m2/" --activation_type m2run1


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = one with method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/onem2/" --activation_type onem2run1



echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = one with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/onem1/" --activation_type onem1run1


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/onethreem1/" --activation_type onethreem1run1


echo "####################### TRAINING FOR GOOGLENET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type GoogLeNet --model_dir "saved/GoogLeNet/onethreem2/" --activation_type onethreem2run1



echo "RESNET50"

echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = ReLu ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/relu/" --activation_type relurun1


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/tanh/" --activation_type tanhrun1


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/sigmoid/" --activation_type sigmoidrun1


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = Adaptive ReLU ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/arelu/" --activation_type arelurun1 


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = Adaptive Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/atanh/" --activation_type atanhrun1

echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = Adaptive sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/asigmoid/" --activation_type asigmoidrun1


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/m1/" --activation_type m1run1

echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/m2/" --activation_type m2run1


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = one with method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/onem2/" --activation_type onem2run1



echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = one with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/onem1/" --activation_type onem1run1


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/onethreem1/" --activation_type onethreem1run1


echo "####################### TRAINING FOR RESNET50 ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type ResNet50 --model_dir "saved/ResNet50/onethreem2/" --activation_type onethreem2run1



echo "################################################### DENSENET ##################################################################"
echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = ReLU  ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type relurun1

echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type tanhrun1


echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type sigmoidrun1


echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = Adaptive ReLU ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type arelurun1


echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = Adaptive Tanh ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type atanhrun1

echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = Adaptive sigmoid ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type asigmoidrun1


echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type m1run1

echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type m2run1


echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = one with method2  ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type onem1run1



echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = one with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type onem2run1


echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type onethreem1run1


echo "####################### TRAINING FOR DENSENET ACTIVATION_TYPE = onethree with method1  ######################################"
CUDA_VISIBLE_DEVICES=0 python Densenet_Cifar10_save.py --activation_type onethreem2run1























