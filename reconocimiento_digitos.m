entradas = readmatrix('entradas.csv');
salidas = readmatrix('salidas.csv');

P = entradas;      
T = salidas;

net = newff([min(P')' max(P')'],[30, 25, 20, 15, 10],{'tansig', 'logsig', 'satlin', 'tansig', 'purelin'},'traingdm');

net.trainParam.epochs = 10000;  
net.trainParam.goal = 0.5e-3;   
net.trainParam.lr = 0.1;        
net.trainParam.mc = 0.4;        

net = train(net, P, T); 

C = sim(net, P); 
