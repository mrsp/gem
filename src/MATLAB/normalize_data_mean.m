function res = normalize_data_mean(data)

res = (data - mean(data))/(max(data) - min(data));

end