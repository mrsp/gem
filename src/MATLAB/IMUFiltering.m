
save_data = 0;
window = 7;
baccY_RLf = sgolayfilt(baccY_RL,3,window);
baccY_LLf = sgolayfilt(baccY_LL,3,window);
baccYf = sgolayfilt(baccY,3,window);

baccX_RLf = sgolayfilt(baccX_RL,3,window);
baccX_LLf = sgolayfilt(baccX_LL,3,window);
baccXf = sgolayfilt(baccX,3,window);

baccZ_RLf = sgolayfilt(baccZ_RL,3,window);
baccZ_LLf = sgolayfilt(baccZ_LL,3,window);
baccZf = sgolayfilt(baccZ,3,window);

if(save_data == 1)
    data_dir = 'C:\Users\stpip\Desktop\gem\GEM2_nao_training\';
    cd(data_dir)
    %LLeg Label
    dlmwrite('baccX_LLf.txt',baccX_LLf)
    dlmwrite('baccY_LLf.txt',baccY_LLf)
    dlmwrite('baccZ_LLf.txt',baccZ_LLf)
    %RLeg Label 
    dlmwrite('baccX_RLf.txt',baccX_RLf)
    dlmwrite('baccY_RLf.txt',baccY_RLf)
    dlmwrite('baccZ_RLf.txt',baccZ_RLf)
    dlmwrite('baccXf.txt',baccXf)
    dlmwrite('baccYf.txt',baccYf)
    dlmwrite('baccZf.txt',baccZf)
end
% fc = 8.0; %15 fsr 10 acc
% fs = 100;
% [b,a] = butter(2,fc/(fs/2));
% gXf = filter(b,a,gX);
% dgXf = diff(gXf);
% dgX = diff(gX);
% 
% 
% figure
% plot(dgX(5000:7000))
% hold on
% plot(dgXf(5000:7000))