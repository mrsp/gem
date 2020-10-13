import numpy as np
import glob


def main():
    #accX 1
    if(glob.glob('./NAO_GEM2/IMU_INTEL/**/accX.txt')):
        files = glob.glob('./NAO_GEM2/IMU_INTEL/**/accX.txt') 
    else:
        print("No files in path")
        return -1

    with open( '../GEM2_nao_training/accX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #accY 2
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/accY.txt')
    with open( '../GEM2_nao_training/accY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #accZ 3
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/accZ.txt')
    with open( '../GEM2_nao_training/accZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    
    #baccX_LL 4
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccX_LL.txt')
    with open( '../GEM2_nao_training/baccX_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_LL 5 
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccY_LL.txt')
    with open( '../GEM2_nao_training/baccY_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_LL 6
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccZ_LL.txt')
    with open( '../GEM2_nao_training/baccZ_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccX_RL 7
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccX_RL.txt')
    with open( '../GEM2_nao_training/baccX_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_RL 8
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccY_RL.txt')
    with open( '../GEM2_nao_training/baccY_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_RL 9
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccZ_RL.txt')
    with open( '../GEM2_nao_training/baccZ_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )            

    #baccX 10
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccX.txt')
    with open( '../GEM2_nao_training/baccX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY 11
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccY.txt')
    with open( '../GEM2_nao_training/baccY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ 12
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccZ.txt')
    with open( '../GEM2_nao_training/baccZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #comvX 13
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/comvX.txt')
    with open( '../GEM2_nao_training/comvX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #comvY 14
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/comvY.txt')
    with open( '../GEM2_nao_training/comvY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #comvZ 15
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/comvZ.txt')
    with open( '../GEM2_nao_training/comvZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lfX 16
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lfX.txt')
    with open( '../GEM2_nao_training/lfX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lfY 17
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lfY.txt')
    with open( '../GEM2_nao_training/lfY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lfZ 18
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lfZ.txt')
    with open( '../GEM2_nao_training/lfZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rfX 19
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rfX.txt')
    with open( '../GEM2_nao_training/rfX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rfY 20
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rfY.txt')
    with open( '../GEM2_nao_training/rfY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rfY 21 
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rfZ.txt')
    with open( '../GEM2_nao_training/rfZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #ltX 22
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/ltX.txt')
    with open( '../GEM2_nao_training/ltX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #ltY 23
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/ltY.txt')
    with open( '../GEM2_nao_training/ltY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #ltZ 24
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/ltZ.txt')
    with open( '../GEM2_nao_training/ltZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rtX 25
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rtX.txt')
    with open( '../GEM2_nao_training/rtX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rtY 26
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rtY.txt')
    with open( '../GEM2_nao_training/rtY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rtZ 27
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rtZ.txt')
    with open( '../GEM2_nao_training/rtZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #laccX 28
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/laccX.txt')
    with open( '../GEM2_nao_training/laccX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #laccY 29
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/laccY.txt')
    with open( '../GEM2_nao_training/laccY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #laccZ 30 
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/laccZ.txt')
    with open( '../GEM2_nao_training/laccZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #raccX 31
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/raccX.txt')
    with open( '../GEM2_nao_training/raccX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #raccY 32
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/raccY.txt')
    with open( '../GEM2_nao_training/raccY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #raccZ 33
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/raccZ.txt')
    with open( '../GEM2_nao_training/raccZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rvX 34
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rvX.txt')
    with open( '../GEM2_nao_training/rvX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rvY 35
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rvY.txt')
    with open( '../GEM2_nao_training/rvY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rvZ 35
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rvZ.txt')
    with open( '../GEM2_nao_training/rvZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           
    #lvX 36
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lvX.txt')
    with open( '../GEM2_nao_training/lvX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lvY 37
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lvY.txt')
    with open( '../GEM2_nao_training/lvY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lvZ 38
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lvZ.txt')
    with open( '../GEM2_nao_training/lvZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           


    #rwX 39
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rwX.txt')
    with open( '../GEM2_nao_training/rwX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rwY 40
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rwY.txt')
    with open( '../GEM2_nao_training/rwY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rwZ 41 
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/rwZ.txt')
    with open( '../GEM2_nao_training/rwZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           
    #lwX 42
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lwX.txt')
    with open( '../GEM2_nao_training/lwX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lwY 43
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lwY.txt')
    with open( '../GEM2_nao_training/lwY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lwZ 44
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/lwZ.txt')
    with open( '../GEM2_nao_training/lwZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )           

    #gX 45
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/gX.txt')
    with open( '../GEM2_nao_training/gX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #gY 46
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/gY.txt')
    with open( '../GEM2_nao_training/gY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #gZ 47
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/gZ.txt')
    with open( '../GEM2_nao_training/gZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )



    #baccX_LL 4
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccXf_LL.txt')
    with open( '../GEM2_nao_training/baccXf_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_LL 5 
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccYf_LL.txt')
    with open( '../GEM2_nao_training/baccYf_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_LL 6
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccZf_LL.txt')
    with open( '../GEM2_nao_training/baccZf_LL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccX_RL 7
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccXf_RL.txt')
    with open( '../GEM2_nao_training/baccXf_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY_RL 8
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccYf_RL.txt')
    with open( '../GEM2_nao_training/baccYf_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ_RL 9
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccZf_RL.txt')
    with open( '../GEM2_nao_training/baccZf_RL.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )            

    #baccX 10
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccXf.txt')
    with open( '../GEM2_nao_training/baccXf.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccY 11
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccYf.txt')
    with open( '../GEM2_nao_training/baccYf.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #baccZ 12
    files = glob.glob('./NAO_GEM2/IMU_INTEL/**/baccZf.txt')
    with open( '../GEM2_nao_training/baccZf.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    return 0

if __name__ == "__main__":
    main()