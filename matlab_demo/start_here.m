   %% ТОКОВАЯ ДИАГНОСТИКА + НЕЙРОМОРФНАЯ ПЛАТФОРМА (АЛТАЙ / KNP) — ДЕМО
% Запускать по секциям: курсор в секцию -> Ctrl+Enter (Cmd+Enter на Mac),
% или всё сразу кнопкой Run.
%
% Данные: 45 испытаний стенда (АД -> муфта -> мультипликатор -> вал на подшипниках
% NSK 6205 -> тормоз), ток статора 3 фазы, 25,6 кГц. Файлы: ../data/mat/current_N.mat
% Нужен запущенный Docker Desktop (для секций 10-13 с нейроморфной платформой).
% РАБОЧАЯ ПОЛОСА нейросети: 0-150 Гц (ток прореживается до 512 Гц). Секции 7-9 (огибающая и
% вейвлеты) дополнительно смотрят ВЧ-часть тока до 12,8 кГц — там ищутся ударные дефекты.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here, fullfile(fileparts(here), 'knp-lab', 'matlab_bridge'));
fs = 25600;                     % частота дискретизации, Гц
FMAX = 150;                     % верхняя граница рабочей полосы, Гц
TRIAL = 1;                      % какое испытание смотреть в секциях 1-2, 5-9 (1..45)

%% 1. КАК ВЫГЛЯДЯТ ДАННЫЕ
% Ток трёх фаз за 0,1 с. Видно ~5 периодов синусоиды — частота питания ~48,5 Гц,
% а не 50: двигатель питается от частотного преобразователя.
% Фаза T во всех записях на ~23% ниже — особенность измерения, не дефект.
% У испытаний 11,15,17,19,21,23,25,27,31 записана только фаза R.
t0 = (0:round(0.1*fs)-1)' / fs;
figure('Name', '1. Сырой ток');
hold on;
for ph = ["R" "S" "T"]
    x = load_trial(TRIAL, ph);
    if ~all(isnan(x)), plot(t0*1000, x(1:numel(t0)), 'DisplayName', "фаза " + ph); end
end
xlabel('Время, мс'); ylabel('Ток, А'); grid on; legend;
title(sprintf('Испытание %d: мгновенный ток статора', TRIAL));

%% 2. СПЕКТР ТОКА (БПФ) — «РАСКЛАДЫВАЕМ ТОК НА НОТЫ»
% 60 с записи -> разрешение 0,0167 Гц. Шкала в дБ относительно основной гармоники
% (0 дБ = сеть). Дефекты механики должны давать слабые «подголоски» вокруг основной
% на частотах, которые считаются от скорости вала.
x = load_trial(TRIAL);
[f, db, amp] = mcsa_spectrum(x, fs, 60);
f_s = peak_near(f, amp, 48.6, 0.6);
f_r1 = (peak_near(f, amp, f_s + 24.2, 0.4) - peak_near(f, amp, f_s - 24.2, 0.4)) / 2;
fprintf('\nИспытание %d: частота сети f_s = %.3f Гц, реальная частота вала f_r1 = %.3f Гц (%.0f об/мин, не 1770 из паспорта)\n', ...
        TRIAL, f_s, f_r1, f_r1*60);

figure('Name', '2. Спектр тока');
subplot(2,1,1);
plot(f, db, 'Color', [.3 .3 .3]); xlim([0 FMAX]); ylim([-110 5]); grid on; hold on;
marks = [f_s, f_s-f_r1, f_s+f_r1, f_s-43.07, f_s+43.07];
names = ["f_s сеть", "f_s−f_{r1}", "f_s+f_{r1}", "помеха 43,07", "помеха 43,07"];
cols = {'k', 'b', 'b', 'm', 'm'};
for i = 1:numel(marks)
    xline(marks(i), ':', names(i), 'Color', cols{i}, 'LabelOrientation', 'aligned');
end
ylabel('дБ отн. основной');
title(sprintf('Испытание %d: спектр в рабочей полосе 0–%d Гц. Синие — вращение вала, фиолетовые — помеха стенда', TRIAL, FMAX));
subplot(2,1,2);
plot(f, db, 'Color', [.3 .3 .3]); xlim([f_s-8 f_s+8]); ylim([-110 5]); grid on;
xlabel('Частота, Гц'); ylabel('дБ');
title('Увеличение около основной гармоники (f_s ± 8 Гц)');

%% 3. РЕАЛЬНЫЕ СКОРОСТИ И РЕЖИМЫ РАБОТЫ ВСЕХ 45 ИСПЫТАНИЙ
% По боковым f_s ± f_r1 меряем скорость вала в каждой записи. По скольжению
% записи делятся на 3 нагрузки, по частоте питания — на 2 варианта: 6 режимов.
% Первый запуск ~1-2 мин (потом берётся из кэша matlab_demo/cache).
T = trial_table();
disp(T(:, {'trial','seconds','f_s','rpm','slip','load_level','supply'}));
disp(groupcounts(T, 'mode'));

figure('Name', '3. Режимы работы');
gscatter(T.slip*100, T.rms_R, T.mode, [], 'o', 8);
text(T.slip*100, T.rms_R, " " + string(T.trial), 'FontSize', 7);
xlabel('Скольжение, %  (нагрузка)'); ylabel('СКО тока фазы R, А'); grid on;
title('Каждая точка — испытание. Кластеры = режимы работы стенда');

%% 4. СРАВНЕНИЕ ИСПЫТАНИЙ ОДНОГО РЕЖИМА — ГЛАВНЫЙ РЕЗУЛЬТАТ АНАЛИЗА
% Берём записи с одинаковой нагрузкой и одинаковой частотой питания и накладываем
% спектры. Итог: они практически совпадают — стандартными признаками токовой
% диагностики испытания внутри режима неразличимы.
grp = T.trial(T.mode == "средняя / номинал")';
figure('Name', '4. Один режим — наложение');
hold on;
for t = grp
    [f2, db2] = mcsa_spectrum(load_trial(t), fs, 60);
    m = f2 <= FMAX;
    plot(f2(m), db2(m), 'DisplayName', sprintf('№%d', t));
end
xlim([0 FMAX]); ylim([-110 5]); grid on; legend('NumColumns', numel(grp));
xlabel('Частота, Гц'); ylabel('дБ отн. основной');
title("Режим «средняя нагрузка / номинал»: испытания " + strjoin(string(grp), ", "));

%% 5. САМЫЕ ГРОМКИЕ ПИКИ В ПОЛОСЕ 0–150 ГЦ — ЧТО ЭТО ТАКОЕ
% Находим 12 самых сильных пиков (кроме основной) и объясняем каждый: гармоника сети
% (k·f_s), вращение вала двигателя (f_s ± k·f_r1), выходного вала (f_s ± k·1,5·f_r1) или
% постоянная помеха стенда (43,07 Гц и её комбинации). Дефектов среди них нет.
bandsel = (f > 2) & (f <= FMAX) & (abs(f - f_s) > 0.5);
[pk, loc] = findpeaks(db(bandsel), f(bandsel), 'MinPeakDistance', 0.5, 'SortStr', 'descend', 'NPeaks', 12);
fprintf('\nСамые сильные пики 0–%d Гц (дБ отн. основной):\n', FMAX);
figure('Name', '5. Пики 0–150 Гц');
plot(f, db, 'Color', [.3 .3 .3]); xlim([0 FMAX]); ylim([-110 5]); grid on; hold on;
for i = 1:numel(loc)
    fp = loc(i);
    k = round(fp / f_s); o1 = (fp - f_s) / f_r1; o2 = (fp - f_s) / (1.5 * f_r1);
    if abs(fp - k*f_s) < 0.1,             what = sprintf('%d·f_s (гармоника сети)', k);
    elseif abs(o1 - round(o1)) < 0.01,    what = sprintf('f_s %+d·f_{r1} (вал двигателя)', round(o1));
    elseif abs(o2 - round(o2)) < 0.01,    what = sprintf('f_s %+d·f_{r2} (выходной вал)', round(o2));
    elseif min(abs(abs(fp - f_s) - [43.07 86.13 28.72])) < 0.1 || min(abs(fp - [43.07 86.13 28.72])) < 0.1
        what = 'помеха стенда';
    else,                                 what = 'не опознан';
    end
    fprintf('  %7.2f Гц  %6.1f дБ  %s\n', fp, pk(i), what);
    plot(fp, pk(i), 'rv', 'MarkerFaceColor', 'r');
end
xlabel('Частота, Гц'); ylabel('дБ');
title('Красные метки — 12 самых сильных пиков в рабочей полосе (расшифровка в Command Window)');

%% 6. ГДЕ ИСКАТЬ ДЕФЕКТЫ ПОДШИПНИКА 6205 (РАСЧЁТ ПО РЕАЛЬНОЙ СКОРОСТИ)
% 6205: 9 шариков, d = 7,94 мм, D = 38,5 мм. Частоты дефектов пропорциональны
% скорости вала; в токе они видны как боковые f_s ± f_деф. Выходной вал по спектру
% вращается в 1,5 раза быстрее вала двигателя.
fr2 = 1.5 * f_r1;
K = struct('FTF', 0.397, 'BSF', 2.322, 'BPFO', 3.572, 'BPFI', 5.428);
names = fieldnames(K);
figure('Name', '6. Частоты подшипника');
plot(f, db, 'Color', [.3 .3 .3]); xlim([0 FMAX]); ylim([-110 5]); grid on; hold on;
fprintf('\nЧастоты дефектов подшипника выходного вала (f_r2 = %.2f Гц), полоса 0-%d Гц:\n', fr2, FMAX);
for i = 1:numel(names)
    Fd = K.(names{i}) * fr2;
    sbs = [abs(f_s - Fd), f_s + Fd];
    inband = sbs(sbs <= FMAX);
    fprintf('  %-5s = %6.1f Гц -> боковые %6.1f и %6.1f Гц; в полосе: %s\n', names{i}, Fd, sbs(1), sbs(2), ...
            mat2str(round(inband, 1)));
    if ~isempty(inband), xline(inband, 'g:', names{i}); end
end
xlabel('Частота, Гц'); ylabel('дБ');
title('Зелёные — где в полосе 0–150 Гц видны дефекты подшипника; явных пиков там нет');

%% 7. СПЕКТР ОГИБАЮЩЕЙ — «КАК ДРОЖИТ АМПЛИТУДА ТОКА»
% Механический дефект модулирует момент, а значит амплитуду тока: i(t) = A(1 + m·g(t))·cos(2π f_s t).
% В прямом спектре это пара боковых f_s ± F. Огибающая |hilbert(i)| — это сразу m·g(t): дефект
% виден на самой частоте F, а высота линии равна глубине модуляции m (0,01 = 1 %).
% (а) Огибающая несущей 48,5 Гц. Одна фаза: при F > f_s линия делится на F и |F − 2f_s| (зеркало
%     через 0 Гц). Вектор Парка из трёх фаз даёт одну линию на F — так чище (если есть 3 фазы).
% (б) Огибающая ВЧ 1–10 кГц — для ударных дефектов. В нашем токе она целиком электрическая:
%     линии 2·f_s, 4·f_s, 6·f_s от ШИМ преобразователя. Пик «BPFI» из чернового анализа — это
%     4·f_s = 194 Гц, а не подшипник.
x = load_trial(TRIAL); x = x(1:min(end, 60*fs));
fr2 = 1.5 * f_r1;
Fb = [0.397 2.322 3.572 5.428] * fr2;
Fn = ["FTF" "BSF" "BPFO" "BPFI"];
[fe1, ae1] = envelope_spectrum(x, fs, [5 400]);
figure('Name', '7. Спектр огибающей');
subplot(3,1,1);
semilogy(fe1, ae1, 'Color', [.2 .4 .8]); xlim([0 300]); ylim([1e-6 1e-1]); grid on; hold on;
xline(Fb, 'r:', Fn); xline([f_r1 fr2 2*fr2], 'b:', ["f_{r1}" "f_{r2}" "2f_{r2}"]); xline(43.07, 'm:', '43,07');
ylabel('глубина модуляции');
title(sprintf('Испытание %d: огибающая несущей, фаза R. Красные — частоты подшипника: линий там нет', TRIAL));
subplot(3,1,2);
X = [load_trial(TRIAL, 'R'), load_trial(TRIAL, 'S'), load_trial(TRIAL, 'T')];
if ~any(isnan(X(:)))
    [fp, ap] = park_envelope_spectrum(X(1:numel(x), :), fs);
    semilogy(fp, ap, 'Color', [.1 .6 .3]); xlim([0 300]); ylim([1e-6 1e-1]); grid on; hold on;
    xline(Fb, 'r:', Fn);
    ylabel('глубина модуляции');
    title('Огибающая вектора Парка (3 фазы): нет зеркальных линий |F − 2f_s|');
else
    text(0.3, 0.5, 'В этом испытании записана только фаза R — вектор Парка не построить', 'Units', 'normalized');
end
subplot(3,1,3);
[fe2, ae2] = envelope_spectrum(x, fs, [1000 10000], true);
plot(fe2, ae2, 'Color', [.3 .3 .3]); xlim([0 400]); grid on; hold on;
xline((1:8) * f_s, 'g--'); xline(Fb, 'r:', Fn);
xlabel('Частота, Гц'); ylabel('амплитуда (отн.)');
title('Огибающая ВЧ 1–10 кГц. Зелёные — k·f_s: всё, что видно, — ШИМ преобразователя, не механика');

%% 8. ВЕЙВЛЕТ-АНАЛИЗ — «СПЕКТР, КОТОРЫЙ МЕНЯЕТСЯ ВО ВРЕМЕНИ»
% (а) Скалограмма CWT (вейвлет Морле) тока 5–300 Гц за 4 с. Ровные горизонтальные полосы = запись
%     стационарна, поэтому 60-секундное БПФ законно (частота сети гуляет всего на ~0,3 мГц).
%     У дефекта, который появляется и пропадает, полоса была бы прерывистой — БПФ этого не видит.
% (б) Вейвлет-пакеты (modwpt, fk18, уровень 5 = 32 полосы по 400 Гц): эксцесс каждой полосы.
%     Высокий эксцесс = импульсы. В нашем токе максимум в 4–4,8 кГц во ВСЕХ 45 записях — это
%     импульсы переключения ШИМ (их линии привязаны к f_s, а не к валу), не дефект.
%     Вывод: полосу для огибающей выбираем не по эксцессу, а по силе линий на частотах дефекта (секция 9).
xd = resample(x(1:4*fs) - mean(x), 1024, fs);
[wt, fw] = cwt(xd, 'amor', 1024, 'FrequencyLimits', [5 300]);
figure('Name', '8. Вейвлет-анализ');
subplot(2,1,1);
scalogram_plot(wt, fw, 1024, 60);
yline([f_s 2*f_s 3*f_s], 'w:');
title(sprintf('Испытание %d: скалограмма тока (Морле). Полосы ровные — сигнал стационарен', TRIAL));
x10 = x(1:10*fs);
[wpt, ~, cf] = modwpt(x10 - mean(x10), 'fk18', 5);
kw = kurtosis(wpt, 1, 2) - 3;
subplot(2,1,2);
bar(cf * fs / 1000, kw, 1, 'FaceColor', [.8 .4 .2]); grid on;
xlabel('Центр полосы, кГц'); ylabel('эксцесс (0 = шум)');
[~, kmax] = max(kw);
title(sprintf('Вейвлет-пакеты, 32 полосы: самая импульсная — около %.1f кГц (ШИМ преобразователя)', cf(kmax) * fs / 1000));

%% 9. ТРИ МЕТОДА НА ОДНОМ ДЕФЕКТЕ + ИТОГ ПРОВЕРКИ
% В 10 с реальной записи добавляем дефект наружного кольца двумя способами:
%   верх — модуляция тока на BPFO (как в синтетике для «Алтая»), уровень −58 дБ («средняя»);
%   низ  — удары на BPFO с 1 % проскальзыванием, звенящие на резонансе 2–6 кГц, пик = ВЧ-шуму.
% Модуляцию ловят прямой спектр и огибающая несущей; удары прямой спектр не видит, огибающая
% «по эксцессу» уходит в полосу ШИМ, а огибающая полосы, выбранной по частотам дефекта, — видит.
x10 = x(20*fs+1:30*fs);
t10 = (0:numel(x10)-1)' / fs;
BPFO = 3.572 * fr2;
m = 2 * 10^(-58/20);
xa = x10 .* (1 + m * (cos(2*pi*BPFO*t10) + 0.5*cos(2*pi*2*BPFO*t10)));
xi = add_impacts(x10, fs, BPFO, 0, 1);
figure('Name', '9. Три метода на одном дефекте', 'Position', [60 60 1500 750]);
[f9, db9] = mcsa_spectrum(xa, fs, 10);
subplot(2,3,1); plot(f9, db9, 'Color', [.3 .3 .3]); xlim([0 300]); ylim([-110 5]); grid on;
xline([abs(f_s - BPFO), f_s + BPFO], 'r:'); title({'Модуляция −58 дБ', 'прямой спектр: пара f_s ± BPFO'}); xlabel('Гц'); ylabel('дБ');
[fa, aa] = envelope_spectrum(xa, fs, [5 400]);
subplot(2,3,2); semilogy(fa, aa); xlim([0 300]); ylim([1e-6 1e-1]); grid on;
xline([BPFO, abs(BPFO - 2*f_s), 2*BPFO], 'r:'); title({'Модуляция', 'огибающая несущей: линии BPFO, |BPFO−2f_s|, 2·BPFO'}); xlabel('Гц');
[wa, fwa] = cwt(envelope_band(xa, fs), 'amor', 1024, 'FrequencyLimits', [5 300]);
subplot(2,3,3); scalogram_plot(wa, fwa, 1024, 50); yline(BPFO, 'r:');
title({'Модуляция', 'скалограмма огибающей: ровная полоса на BPFO'});
[f9, db9] = mcsa_spectrum(xi, fs, 10);
subplot(2,3,4); plot(f9, db9, 'Color', [.3 .3 .3]); xlim([0 8000]); ylim([-140 5]); grid on;
title({'Удары (пик = ВЧ-шуму)', 'прямой спектр: линий BPFO нет'}); xlabel('Гц'); ylabel('дБ');
[fk, ak] = envelope_spectrum(xi, fs, [4000 4800], true);
subplot(2,3,5); plot(fk, ak); xlim([0 400]); grid on; xline((1:3) * BPFO, 'r:'); xline((1:8) * f_s, 'g--');
title({'Удары', 'полоса «по эксцессу» 4–4,8 кГц: линии ШИМ (k·f_s) сильнее BPFO'}); xlabel('Гц');
[band, snr9, fb, ab] = best_band(xi, fs, [BPFO 2*BPFO]);
subplot(2,3,6); plot(fb, ab); xlim([0 400]); grid on; xline((1:3) * BPFO, 'r:');
title({'Удары', sprintf('полоса %.0f–%.0f Гц выбрана по BPFO: SNR %.0f дБ', band, snr9)}); xlabel('Гц');

% Итог проверки на синтетике (analysis/compare_methods.py): 45 записей, порог по чистым кускам.
root = fileparts(here);
C = readtable(fullfile(root, 'analysis', 'compare_methods.csv'), 'TextType', 'string');
C = C(C.family ~= "normal", :);
G = groupsummary(C, {'family', 'severity', 'method'}, 'mean', 'detected');
G = unstack(G(:, {'family', 'severity', 'method', 'mean_detected'}), 'mean_detected', 'method');
G{:, 3:end} = round(100 * G{:, 3:end});
disp('Обнаружено, % (am — модуляция, impact — удары; severity 1..3 — слабая..сильная):');
disp(G);

%% 10. НЕЙРОМОРФНАЯ ЧАСТЬ: ТОК -> СПАЙКИ -> СПАЙКОВАЯ СЕТЬ НА KNP -> ОТВЕТ
% Платформа Kaspersky Neuromorphic Platform (под неё работает чип «Алтай») запущена
% в Docker. Цепочка как в будущем изделии:
%   1) delta-кодер: событие (спайк) когда ток вырос больше порога с прошлого события;
%   2) спайки уходят в сеть KNP (через CSV-мост в контейнер knp-run);
%   3) сеть возвращает выходные спайки обратно в MATLAB.
% Бэкенд — CPU KNP v2 (демонстрация моста; эмулятор чипа «Алтай» — в секциях 11-13).
x = load_trial(TRIAL);
N = 60;                                     % 60 шагов сети = 60 мс сигнала
xs = x(1:round(fs/1000):end);               % ток с шагом 1 мс
xs = xs(1:N);
thr = 0.35 * std(xs);
in_steps = [];
last = xs(1);
for k = 2:N
    if xs(k) - last > thr
        in_steps(end+1) = k - 1; %#ok<AGROW>
        last = xs(k);
    elseif xs(k) < last
        last = xs(k);
    end
end
out_steps = knp_run_network(in_steps, N, 'cpu');

figure('Name', '7. Нейроморфная цепочка');
subplot(3,1,1); plot(0:N-1, xs, '.-'); grid on; ylabel('Ток, А');
title('Ток (шаг 1 мс) -> delta-кодер -> спайковая сеть KNP -> выходные спайки');
subplot(3,1,2); stem(in_steps, ones(size(in_steps)), 'b', 'Marker', 'none', 'LineWidth', 1.5);
xlim([0 N-1]); ylim([0 1.2]); yticks([]); ylabel('вход');
subplot(3,1,3); stem(out_steps, ones(size(out_steps)), 'r', 'Marker', 'none', 'LineWidth', 1.5);
xlim([0 N-1]); ylim([0 1.2]); yticks([]); ylabel('выход'); xlabel('Шаг сети (мс)');
fprintf('\nВходных спайков: %d, выходных: %d\n', numel(in_steps), numel(out_steps));

%% 11. ЭМУЛЯТОР ЧИПА «АЛТАЙ» (AltAI golden model, KNP v1.0.0)
% Бэкенд AltAI из релиза v2.0.0 падает при загрузке любой сети. Бэкенд из v1.0.0 работает после
% одной правки ядра: пакет AltAI собран против ядра с методом Backend::select_device, которого нет
% в публичном теге v1.0.0 (таблица виртуальных методов была сдвинута). Метод перенесён из v2.
% Тест: 1 вход -> 1 нейрон AltAILIF, входные спайки на шагах 0, 2, 3 -> выход на тех же шагах.
[~, out] = system(['/usr/local/bin/docker exec knp-lab bash -c ' ...
    '"cd /work/repo-v1/build/bin && ./altai-fault-classifier /opt/altai-v1/usr/lib/knp-altai-backend ' ...
    '/work/chip_io/t1_w.csv /work/chip_io/t1_in.csv /work/chip_io/t1_out.csv --select-device 2>&1 | grep -v placer_logger; ' ...
    'echo --- выходные спайки шаг,нейрон:; cat /work/chip_io/t1_out.csv"']);
disp(out);

%% 12. СИНТЕТИЧЕСКИЕ ДЕФЕКТЫ: ОДИН ФАЙЛ -> ДИАГНОЗ «АЛТАЯ»
% В реальные записи 31-45 впрыснуты симптомы 6 дефектов x 3 степени (модуляция тока на
% частоте дефекта). Результат — 45 файлов в формате преподавателя: ../data/synthetic/csv/syn_NN.csv,
% ключ ответов — ../data/synthetic/labels.csv. Сеть обучена на записях 1-30 и этих файлов не видела.
% Цепочка: 6 кусков по 10 с -> прореживание до 512 Гц -> 7 признаков в полосе 0-150 Гц
% (дБ над нормой) -> 42 бинарных спайка -> 9 нейронов AltAILIF (веса -1/0/1, 60 синапсов) ->
% решение каждые 10 с -> голосование за минуту.
SYN = 'syn_07.csv';           % любой из syn_01..syn_45
BACKEND = 'altai-v1';         % 'altai-v1' — эмулятор чипа AltAI v1.0.0; 'cpu-v2' — нейрон «Алтая» в KNP v2
root = fileparts(here);
L = readtable(fullfile(root, 'data', 'synthetic', 'labels.csv'), 'TextType', 'string');
row = L(L.file == SYN, :);
M = readmatrix(fullfile(root, 'data', 'synthetic', 'csv', SYN));
[f9, db9, amp9] = mcsa_spectrum(M(:, 1), fs, 60);
fs9 = peak_near(f9, amp9, 48.6, 0.6);
fr1_9 = (peak_near(f9, amp9, fs9 + 24.2, 0.4) - peak_near(f9, amp9, fs9 - 24.2, 0.4)) / 2;
fr2_9 = 1.5 * fr1_9;
Fdef = struct('outer_race', 3.572*fr2_9, 'inner_race', 5.428*fr2_9, 'ball', 2*2.322*fr2_9, ...
              'cage', 0.397*fr2_9, 'unbalance', fr2_9, 'misalignment', 2*fr2_9, 'normal', NaN);
F = Fdef.(char(row.defect));
figure('Name', '9. Синтетический дефект');
plot(f9, db9, 'Color', [.3 .3 .3]); grid on; hold on;
if ~isnan(F)
    sbs = [abs(fs9 - F), fs9 + F];
    sbs = sbs(sbs <= FMAX);   % видим только то, что попадает в рабочую полосу
    if ~isempty(sbs), xline(sbs, 'r--', 'симптом', 'LineWidth', 1.2); end
end
xlim([0 FMAX]); ylim([-110 5]); xlabel('Частота, Гц'); ylabel('дБ отн. основной');
title(sprintf('%s: впрыснуто «%s», степень «%s» (красные линии — симптом в полосе 0–%d Гц)', ...
      SYN, row.defect_ru, row.severity_ru, FMAX));
py = fullfile(root, '.venv', 'bin', 'python');
[~, out] = system(sprintf('cd "%s" && "%s" diagnose_file.py %s %s', ...
                  fullfile(root, 'analysis', 'synth'), py, SYN, BACKEND));
disp(out);

%% 13. ИТОГ ПО 45 «СЛЕПЫМ» ФАЙЛАМ
% Результат прогона всех 45 файлов на нейроморфном бэкенде (analysis/synth/infer_blind.py).
R = readtable(fullfile(root, 'data', 'synthetic', sprintf('blind_results_%s.csv', BACKEND)), 'TextType', 'string');
fprintf('Тип дефекта верно: %d из %d (%.0f%%)\n', sum(R.type_ok == "True"), height(R), 100*mean(R.type_ok == "True"));
figure('Name', '10. Матрица ошибок');
confusionchart(categorical(R.truth), categorical(R.pred));
title(sprintf('Диагноз «Алтая» (%s) по 45 файлам: строки — правда, столбцы — ответ сети', BACKEND));
