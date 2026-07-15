% 1. Define sample netlist (e.g., a simple RLC circuit)
netlist_raw = {
    '.title Mmem';
    '.subckt zhangMemristor TE BE w';
    'BR_behavioral TE BE i=(V(TE) - V(BE)) / (1000.0 * (V(w)/3) + 100000.0 * (1.0 - (V(w)/3)))';
    'CC_state w 0 1.5275251633038964e-11 ic=2.5';
    'BGstate 0 w i=( (V(TE) - V(BE)) > 0.3 ) ? ( 6.546537e-06 * (1000.0/3) * (9e-05 / (((V(TE) - V(BE)) / (1000.0 * (V(w)/3) + 100000.0 * (1.0 - (V(w)/3)))) - 1e-06 + 1e-12)) * (1.0 - pow((2.0 * (V(w) / 3) - 1.0), 4)) ) : ( ( (V(TE) - V(BE)) < -0.3 ) ? ( 6.546537e-06 * (1000.0/3) * (((V(TE) - V(BE)) / (1000.0 * (V(w)/3) + 100000.0 * (1.0 - (V(w)/3)))) / 3e-05) * (1.0 - pow((2.0 * (V(w) / 3) - 1.0), 4)) ) : 0.0 )';
    '.ends zhangMemristor';
    'VD D 0 PWL(0s 0V  1e-06s 1V 0.00025s 1V 0.00025100000000000003s 0V 0.0005s 0V 0.000501s 1V 0.00075s 1V 0.000751s 0V 0.001s 0V 0.001001s 1V 0.00125s 1V 0.001251s 0V 0.0015s 0V 0.001501s 1V 0.00175s 1V 0.001751s 0V 0.002s 0V 0.002001s 1V 0.0022500000000000003s 1V 0.0022510000000000004s 0V 0.0025s 0V 0.002501s 1V 0.00275s 1V 0.002751s 0V 0.003s 0V 0.003001s 1V 0.0032500000000000003s 1V 0.0032510000000000004s 0V 0.0035s 0V 0.003501s 1V 0.00375s 1V 0.003751s 0V 0.004s 0V 0.004001s 1V 0.00425s 1V 0.0042510000000000004s 0V 0.0045000000000000005s 0V 0.004501000000000001s 1V 0.004750000000000001s 1V 0.004751000000000001s 0V 0.005000000000000001s 0V 0.005001000000000001s -1V 0.005250000000000001s -1V 0.005251000000000001s 0V 0.005500000000000001s 0V 0.0055010000000000015s -1V 0.005750000000000002s -1V 0.005751000000000002s 0V 0.006000000000000002s 0V 0.006001000000000002s -1V 0.006250000000000002s -1V 0.006251000000000002s 0V 0.006500000000000002s 0V 0.0065010000000000024s -1V 0.0067500000000000025s -1V 0.006751000000000003s 0V 0.007000000000000003s 0V 0.007001000000000003s -1V 0.007250000000000003s -1V 0.007251000000000003s 0V 0.007500000000000003s 0V 0.007501000000000003s -1V 0.007750000000000003s -1V 0.0077510000000000035s 0V 0.008000000000000004s 0V 0.008001000000000003s -1V 0.008250000000000004s -1V 0.008251000000000003s 0V 0.008500000000000004s 0V 0.008501000000000003s -1V 0.008750000000000004s -1V 0.008751000000000004s 0V 0.009000000000000005s 0V 0.009001000000000004s -1V 0.009250000000000005s -1V 0.009251000000000004s 0V 0.009500000000000005s 0V 0.009501000000000004s -1V 0.009750000000000005s -1V 0.009751000000000004s 0V)';
    'Xmmem1 D 0 w_state zhangMemristor';
    '.tran 1000'
};


% Alternatively, if saved to a file, uncomment below:
% netlist_raw = regexp(fileread('circuit.net'), '\n', 'split')';

% 2. Initialize connection tracking arrays
sources = {};
targets = {};
edge_labels = {};

% 3. Parse active top-level topology lines
in_subcircuit = false;

for i = 1:length(netlist_raw)
    line = strtrim(netlist_raw{i});
    if isempty(line), continue; end

    % Track if we are inside a subcircuit definition block to ignore internal details
    if strncmpi(line, '.subckt', 7)
        in_subcircuit = true;
        continue;
    elseif strncmpi(line, '.ends', 5)
        in_subcircuit = false;
        continue;
    end

    % Skip simulation commands, titles, or code blocks within subcircuits
    if in_subcircuit || startsWith(line, '.') || startsWith(line, '*')
        continue;
    end

    % Split the line into component tokens
    tokens = strsplit(line);
    comp_name = tokens{1};
    comp_type = upper(comp_name(1));

    switch comp_type
        case 'V' % 2-Terminal Voltage Source: [Name, Node+, Node-, Type/Value...]
            sources{end+1}     = tokens{2};
            targets{end+1}     = tokens{3};
            edge_labels{end+1} = comp_name;

        case 'X' % Multi-terminal Subcircuit Instance: [Name, Nodes..., SubcktName]
            % For 'Xmmem1 D 0 w_state zhangMemristor':
            % Nodes are at indexes 2, 3, and 4. Index 5 is the model template name.
            subckt_nodes = tokens(2:end-1);
            primary_node = subckt_nodes{1}; % Pin 1 ('D')

            % Graph terminal links from Pin 1 to all other pins to show the network branch
            for n = 2:length(subckt_nodes)
                sources{end+1}     = primary_node;
                targets{end+1}     = subckt_nodes{n};
                edge_labels{end+1} = sprintf('%s (Pin1-%d)', comp_name, n);
            end
    end
end

% 4. Create and render the connectivity graph
G = digraph(sources, targets, table(edge_labels', 'VariableNames', {'Label'}));

figure('Color', 'w');
p = plot(G, 'EdgeLabel', G.Edges.Label, ...
           'NodeLabel', G.Nodes.Name, ...
           'Layout', 'layered', ...
           'MarkerSize', 9, ...
           'NodeFontSize', 11, ...
           'EdgeFontSize', 9);

% 5. Stylize nodes and lines for standard electronics contrast
p.NodeColor = [0.1 0.4 0.7];
p.EdgeColor = [0.3 0.3 0.3];
p.LineWidth = 1.5;
title('Top-Level Circuit Netlist Diagram', 'FontSize', 13);
axis off;

