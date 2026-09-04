%% 1. CARGA DE DATOS
entradas = readmatrix('entradas.csv');   % 20 x 10
salidas  = readmatrix('salidas.csv');    % 10 x 10
P = entradas;
T = salidas;

%% 2. CREACIÓN Y CONFIGURACIÓN DE LA RED
% Arquitectura profunda: 5 capas ocultas [30 25 20 15 10]
net = newff([min(P')' max(P')'], [30, 25, 20, 10], ...
    {'tansig', 'logsig', 'satlin','purelin'}, 'traingdm');
net.trainParam.epochs = 10000;
net.trainParam.goal   = 0.5e-6;
net.trainParam.lr     = 0.1;
net.trainParam.mc     = 0.4;

%% 3. ENTRENAMIENTO
net = train(net, P, T);

%% 4. EVALUACIÓN DE PRECISIÓN
C = sim(net, P);                     % predicción de la red
[~, reales]    = max(T, [], 1);      % dígito real
[~, obtenidos] = max(C, [], 1);      % dígito predicho
reales = reales - 1; obtenidos = obtenidos - 1;  % de índice 1-10 a dígito 0-9

fprintf('\nReal | Obtenido\n-----------------\n');
for i = 1:10
    fprintf(' %d   |   %d\n', reales(i), obtenidos(i));
end
precision = 100 * sum(reales == obtenidos) / 10;
fprintf('-----------------\nPrecisión: %.1f%%\n\n', precision);

%% 6. PRUEBAS DE ROBUSTEZ Y GENERALIZACIÓN (versión corregida)
% Se evalúa la red entrenada (sin reentrenar) ante patrones con ruido creciente.
% El ruido se acumula: en cada paso se invierte UNA celda más (además de las
% ya invertidas), y la prueba se detiene en cuanto la red se equivoca.

digitos  = [3, 5, 8];      % Dígitos a probar
columnas = [4, 6, 9];      % Columna correspondiente en P para cada dígito
maxRuido = 3;              % Tope de celdas a invertir, por si nunca falla
filasImg = 5;
colsImg  = 4;

fprintf('  PRUEBA PROGRESIVA DE ROBUSTEZ\n');

for i = 1:length(digitos)
    digitoEsperado = digitos(i);
    original = P(:, columnas(i));
    nPixeles = length(original);

    fprintf('\n--- Dígito %d ---\n', digitoEsperado);

    % Orden fijo de celdas a ir invirtiendo (se genera UNA sola vez por dígito)
    rng(digitoEsperado);
    ordenCeldas = randperm(nPixeles);

    puntoFallo = NaN;

    for nCeldas = 1:maxRuido
        idx = ordenCeldas(1:nCeldas);   % celdas acumuladas hasta el paso actual

        ruido = original;
        ruido(idx) = 1 - ruido(idx);

        salida = sim(net, ruido);
        [~, reconocido] = max(salida);
        reconocido = reconocido - 1;

        exito = (reconocido == digitoEsperado);
        estado = 'ÉXITO';
        if ~exito
            estado = 'FALLO';
        end

        fprintf('  Celdas alteradas: %d | Índices: %s | Reconocido: %d -> %s\n', ...
            nCeldas, mat2str(idx), reconocido, estado);

        figure;
        imagesc(reshape(ruido, filasImg, colsImg));
        colormap(gray);
        axis equal tight;
        title(sprintf('Dígito %d con %d celdas invertidas (%s)', ...
            digitoEsperado, nCeldas, estado));

        if ~exito
            puntoFallo = nCeldas;
            break;   
        end
    end

    if isnan(puntoFallo)
        fprintf('  >> La red no falló ni con %d celdas alteradas.\n', maxRuido);
    else
        fprintf('  >> La red deja de reconocer el dígito a partir de %d celdas alteradas.\n', puntoFallo);
    end
end
fprintf('=================================\n');