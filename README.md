# MLP기반의 딥러닝 iris 데이터로 학습하기 (Pytorch)
##### -> 내가 구현한 Deep Learning이 맞게 구현했는지, 잘 학습하는지를 보기 위해 많이 쓰는데이터인 iris데이터로 돌려보았다.

***

### <span style="color: blue">코드설명</span>

#### 1. 데이터 불러오기 및 전처리
* numpy를 이용하여 `.dat`파일의 데이터를 불러오고
* 이 데이터를 class가 없기 때문에 for문으로 class를 지정하였다.
* torch에서 신경망에 데이터가 들어가기 위해 tensor 형태로 들어가야 학습이 된다. 그래서 `FloatTensor()`를 사용하였다.

```py
train = np.loadtxt("data/training.dat")
train=torch.FloatTensor(train)

train_X = train
train_Y = []
for i in range(len(train_X)):
    if i >= 0 and i < 25:
        train_Y.append(0)
    elif i >= 25 and i<50:
        train_Y.append(1)
    else:
        train_Y.append(2)
train_Y = torch.tensor(train_Y)
train_Y = F.one_hot(train_Y)#OneHotEncoding
train_Y = train_Y.type(torch.FloatTensor)
...
Train_Data=[]
for i in range(len(train)):
    Train_Data.append((train_X[i],train_Y[i]))

```
<br/>

#### 2. 신경망 구성
* 딥러닝으로 구성하기 위해 Hidden layer를 2개로 지정하였다.
* `BatchNormalization`과 `Dropout`을 사용하여 최적의 학습을 하려고 하였으나 데이터의 양이 적고 모델이 복잡하지 않아 사용을 하진 않았다.
* 활성화 함수로 `ReLU()`와 예측 모델에서 사용되는 `Softmax()`를 사용하였다.
```py
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #input: sample의 size  hidden: output의 size
        super(NeuralNet, self).__init__()
        self.input_layer  = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):        
        output = self.input_layer(x)
        output = self.relu(self.hidden_layer1(output))
        output = self.relu(self.hidden_layer2(output))
        output = self.output_layer(output)
        output = self.soft(output)
        return output
```

* 신경망의 input, hidden, output size를 각각 지정하고
* 손실함수로 `CrossEntropyLoss()`를 사용하였다. (다중 분류 문제에서 성능이 가장 좋다.)
```py
model = NeuralNet(4,16,3)
learning_rate=0.01 #학습율 설정(수정해서 돌려보기)
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
```
<br/>

#### 3. 모델 학습
* 지정된 epoch만큼 반복을 하며 학습을 한다.
* `optimizer.zero_grad()`는 기울기를 초기화하는 것이다. 초기화를 안 하면 기울기가 다른 방향으로 가르켜 학습이 원하는 방향으로 이루어지지 않는다.
* `torch.argmax()`는 배열중 가장 높은 인텍스를 찾는 것이다. 모델이 학습을 하면 output으로 입력된 배열 만큼 0~1 사이의 값이 나온다 그 중 가장 높은 값을 갖는 배열이 모델이 에측한 class의 값이다. 
```py
for epoch in range(training_epochs):
    avg_cost = 0
    count = 0
    list_train_x=[]
    list_train_y=[]
    total_batch = len(data_loader)

    for X,label in data_loader:
        optimizer.zero_grad()
        train_output=model(X)
        cost = criterion(train_output, label)
        cost.backward()
        optimizer.step()
        
        arg_train_x = torch.argmax(train_output,1)
        arg_train_y = torch.argmax(label,1)
   
        list_train_x += arg_train_x.tolist()
        list_train_y += arg_train_y.tolist()
        avg_cost += cost / total_batch 
```
<br/>

#### 4. 모델 테스트
* 학습이 아니고 테스트이므로 기울기 계산을 하면 안 되기 때문에 `torch.no_grad()`를 사용한다.
* `.eval()`은 `Dropout`, `Batchnorm`등을 비활성시켜 추론 모드로 작동한다
* `model()`에서 확습된 모델을 불러와 test 데이터에 대한 예측 값을 불러온다.
* 예측 값과 실제 값과 비교하여 정확도를 게산한다.
```py
with torch.no_grad():
    model.eval()
    test_count = 0
    test_avg_cost = 0
    list_test_x=[]
    list_test_y=[]
    correct = 0
        
    prediction = model(test_x)
    correct_cost = criterion(prediction, test_y)

    arg_test_x = torch.argmax(prediction,1)
    print(arg_test_x)
    arg_test_y = torch.argmax(test_y,1)
    print(arg_test_y)

    list_test_x += arg_test_x.tolist()
    list_test_y += arg_test_y.tolist()
    
    test_avg_cost += correct_cost / total_batch 
```