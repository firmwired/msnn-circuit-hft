% Copy-paste your netlist text inside this multi-line string variable
netlistText = [
    ".title Mmem"
    ".subckt zhangMemristor TE BE w"
    "BR_behavioral TE BE i=(V(TE) - V(BE)) / (1000.0 * (V(w)/3) + 100000.0 * (1.0 - (V(w)/3)))"
    "CC_state w 0 1.5275251633038964e-11 ic=2.5"
    "BGstate 0 w i=( (V(TE) - V(BE)) > 0.3 ) ? ( 6.546537e-06 * (1000.0/3) * (9e-05 / (((V(TE) - V(BE)) / (1000.0 * (V(w)/3) + 100000.0 * (1.0 - (V(w)/3)))) - 1e-06 + 1e-12)) * (1.0 - pow((2.0 * (V(w) / 3) - 1.0), 4)) ) : ( ( (V(TE) - V(BE)) < -0.3 ) ? ( 6.546537e-06 * (1000.0/3) * (((V(TE) - V(BE)) / (1000.0 * (V(w)/3) + 100000.0 * (1.0 - (V(w)/3)))) / 3e-05) * (1.0 - pow((2.0 * (V(w) / 3) - 1.0), 4)) ) : 0.0 )"
    ".ends zhangMemristor"
    "VD D 0 PWL(0s 0V  1e-06s 1V 0.00025s 1V 0.00025100000000000003s 0V 0.0005s 0V 0.000501s 1V 0.00075s 1V 0.000751s 0V 0.001s 0V 0.001001s 1V 0.00125s 1V 0.001251s 0V 0.0015s 0V 0.001501s 1V 0.00175s 1V 0.001751s 0V 0.002s 0V 0.002001s 1V 0.0022500000000000003s 1V 0.0022510000000000004s 0V 0.0025s 0V 0.002501s 1V 0.00275s 1V 0.002751s 0V 0.003s 0V 0.003001s 1V 0.0032500000000000003s 1V 0.0032510000000000004s 0V 0.0035s 0V 0.003501s 1V 0.00375s 1V 0.003751s 0V 0.004s 0V 0.004001s 1V 0.00425s 1V 0.0042510000000000004s 0V 0.0045000000000000005s 0V 0.004501000000000001s 1V 0.004750000000000001s 1V 0.004751000000000001s 0V 0.005000000000000001s 0V 0.005001000000000001s -1V 0.005250000000000001s -1V 0.005251000000000001s 0V 0.005500000000000001s 0V 0.005501000000000001s -1V 0.005750000000000002s -1V 0.005751000000000002s 0V 0.006000000000000002s 0V 0.006001000000000002s -1V 0.006250000000000002s -1V 0.006251000000000002s 0V 0.006500000000000002s 0V 0.0065010000000000024s -1V 0.006750000000000002s -1V 0.006751000000000003s 0V 0.007000000000000003s 0V 0.007001000000000003s -1V 0.007250000000000003s -1V 0.007251000000000003s 0V 0.007500000000000003s 0V 0.007501000000000003s -1V 0.007750000000000003s -1V 0.0077510000000000035s 0V 0.008000000000000004s 0V 0.008001000000000003s -1V 0.008250000000000004s -1V 0.008251000000000003s 0V 0.008500000000000004s 0V 0.008501000000000003s -1V 0.008750000000000004s -1V 0.008751000000000004s 0V 0.009000000000000005s 0V 0.009001000000000004s -1V 0.009250000000000005s -1V 0.009251000000000004s 0V 0.009500000000000005s 0V 0.009501000000000004s -1V 0.009750000000000005s -1V 0.009751000000000004s 0V)"
    "Xmmem1 D 0 w_state zhangMemristor"
    ".tran 1000"
];

% Initialize edge lists for Main Circuit and Subcircuit
mainSrc = {}; mainTgt = {}; mainLbl = {};
subSrc  = {}; subTgt  = {}; subLbl  = {};

inSubckt = false;

for i = 1:length(netlistText)
    line = strtrim(netlistText(i));
    if isempty(line) || startsWith(line, ".") && ~startsWith(line, ".subckt", 'IgnoreCase', true) && ~startsWith(line, ".ends", 'IgnoreCase', true)
        continue; % Skip titles, sim directives, etc.
    end

    parts = strsplit(line);
    cmd = parts{1};

    if strcmpi(cmd, '.subckt')
        inSubckt = true;
        continue;
    elseif strcmpi(cmd, '.ends')
        inSubckt = false;
        continue;
    end

    % Track component node mappings
    if inSubckt
        subSrc{end+1} = parts{2};
        subTgt{end+1} = parts{3};
        subLbl{end+1} = parts{1};
    else
        if startsWith(cmd, 'X', 'IgnoreCase', true)
            % X-elements (subcircuits) map external connections sequentially
            % Xmmem1 maps: D->TE, 0->BE, w_state->w
            mainSrc{end+1} = parts{2}; mainTgt{end+1} = parts{4}; mainLbl{end+1} = [cmd ' (TE to w)'];
            mainSrc{end+1} = parts{2}; mainTgt{end+1} = parts{3}; mainLbl{end+1} = [cmd ' (TE to BE)'];
            mainSrc{end+1} = parts{3}; mainTgt{end+1} = parts{4}; mainLbl{end+1} = [cmd ' (BE to w)'];
        else
            mainSrc{end+1} = parts{2};
            mainTgt{end+1} = parts{3};
            mainLbl{end+1} = parts{1};
        end
    end
end

% Create MATLAB Digraph Objects
G_main = digraph(mainSrc, mainTgt, table(mainLbl', 'VariableNames', {'Label'}));
G_sub  = digraph(subSrc, subTgt, table(subLbl', 'VariableNames', {'Label'}));

% Plot Layout
figure('Position', [100, 100, 1000, 450]);

% Main System View
subplot(1, 2, 1);
p1 = plot(G_main, 'EdgeLabel', G_main.Edges.Label, 'NodeLabel', G_main.Nodes.Name, 'Layout', 'force');
p1.MarkerSize = 8; p1.NodeColor = 'g'; p1.EdgeColor = [0.2 0.2 0.2];
title('Main Circuit Connectivity');
grid on;

% Subcircuit Internal View
subplot(1, 2, 2);
p2 = plot(G_sub, 'EdgeLabel', G_sub.Edges.Label, 'NodeLabel', G_sub.Nodes.Name, 'Layout', 'circle');
p2.MarkerSize = 8; p2.NodeColor = 'm'; p2.EdgeColor = 'b';
title('Subcircuit (zhangMemristor) Details');
grid on;

