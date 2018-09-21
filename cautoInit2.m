function p = cautoInit2(num_conv, extra_hidden, patches, nt, nf)
    % Initialize convolutional auto-encoder weights via Xavier initialization.
    % For a description of parameters, see cautoCost2.m.
  
    num_hidden = patches(1).Nx*patches(2).Ny;
    ndim = nt*nf;

    % initialize convolution layer
    W = randn(num_conv, num_hidden+1) * 2/(num_conv+num_hidden+1);
    n_weights = numel(W);

    if isempty(extra_hidden) || extra_hidden(1) <= 0
        % no extra hidden layers, skip to initialization of reconstruction layer
        R = randn(ndim, num_conv*numel(patches)+1) * 2/(ndim+num_conv*numel(patches)+1);
	n_weights = n_weights + numel(R);

    else
        % initialize extra hidden layers
        extra_layers = extra_hidden(1);
	extra_num = extra_hidden(2);
	H = cell(1, extra_layers);
        H{1} = randn(extra_num, num_conv*length(patches)+1) * 2/(extra_num+num_conv*length(patches)+1);
        n_weights = n_weights + numel(H{1});
	if extra_layers > 1
            for j=2:extra_layers
	        H{j} = randn(extra_num, extra_num+1) * 2/(extra_num*(extra_num+1));
		n_weights = n_weights + numel(H{j});
	    end
	end
	R = randn(ndim, extra_num+1) * 2/(ndim*(extra_num+1));
	n_weights = n_weights + numel(R);
    end

    p = zeros(1, n_weights);
    index = 1:numel(W);
    p(index) = W(:);
    if ~isempty(extra_hidden) && extra_hidden(1) > 0
        for j=1:extra_hidden(1)
            index = index(end)+1:index(end)+numel(H{j});
	    p(index) = H{j}(:);
	end
    end
    index = index(end)+1:index(end)+numel(R);
    p(index) = R(:);
