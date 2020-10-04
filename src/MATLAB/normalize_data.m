function res = normalize_data(data,min_,max_)

res = min_ + (max_ - min_) * (data - min(data))/(max(data) - min(data));

end