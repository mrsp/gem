import numpy as np
import glob


def main():


    #accX 1
    if(glob.glob('../GEM2_talos_training/**/accX.txt')):
        files = glob.glob('../GEM2_talos_training/**/accX.txt')
    else:
        print("No files in path")
        return -1

    with open( '../GEM2_talos_training/accX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #gt
    if(glob.glob('../GEM2_talos_training/**/gt.txt')):
        files = glob.glob('../GEM2_talos_training/**/gt.txt')
        with open( '../GEM2_talos_training/gt.txt', 'w' ) as result:
            for file_ in files:
                for line in open( file_, 'r' ):
                    result.write( line )
    else:
        print("No GT in path")


    #accY 2
    files = glob.glob('../GEM2_talos_training/**/accY.txt')
    with open( '../GEM2_talos_training/accY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #accZ 3
    files = glob.glob('../GEM2_talos_training/**/accZ.txt')
    with open( '../GEM2_talos_training/accZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
  

    #comvX 4
    files = glob.glob('../GEM2_talos_training/**/comvX.txt')
    with open( '../GEM2_talos_training/comvX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #comvY 5
    files = glob.glob('../GEM2_talos_training/**/comvY.txt')
    with open( '../GEM2_talos_training/comvY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #comvZ 6
    files = glob.glob('../GEM2_talos_training/**/comvZ.txt')
    with open( '../GEM2_talos_training/comvZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lfX 7
    files = glob.glob('../GEM2_talos_training/**/lfX.txt')
    with open( '../GEM2_talos_training/lfX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #lfY 8
    files = glob.glob('../GEM2_talos_training/**/lfY.txt')
    with open( '../GEM2_talos_training/lfY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #lfZ 9
    files = glob.glob('../GEM2_talos_training/**/lfZ.txt')
    with open( '../GEM2_talos_training/lfZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rfX 10
    files = glob.glob('../GEM2_talos_training/**/rfX.txt')
    with open( '../GEM2_talos_training/rfX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rfY 11
    files = glob.glob('../GEM2_talos_training/**/rfY.txt')
    with open( '../GEM2_talos_training/rfY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rfY 12 
    files = glob.glob('../GEM2_talos_training/**/rfZ.txt')
    with open( '../GEM2_talos_training/rfZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #ltX 13
    files = glob.glob('../GEM2_talos_training/**/ltX.txt')
    with open( '../GEM2_talos_training/ltX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #ltY 14
    files = glob.glob('../GEM2_talos_training/**/ltY.txt')
    with open( '../GEM2_talos_training/ltY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #ltZ 15
    files = glob.glob('../GEM2_talos_training/**/ltZ.txt')
    with open( '../GEM2_talos_training/ltZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rtX 16
    files = glob.glob('../GEM2_talos_training/**/rtX.txt')
    with open( '../GEM2_talos_training/rtX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #rtY 17
    files = glob.glob('../GEM2_talos_training/**/rtY.txt')
    with open( '../GEM2_talos_training/rtY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #rtZ 18
    files = glob.glob('../GEM2_talos_training/**/rtZ.txt')
    with open( '../GEM2_talos_training/rtZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

 
    #gX 19
    files = glob.glob('../GEM2_talos_training/**/gX.txt')
    with open( '../GEM2_talos_training/gX.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

    #gY 20
    files = glob.glob('../GEM2_talos_training/**/gY.txt')
    with open( '../GEM2_talos_training/gY.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )
    #gZ 21
    files = glob.glob('../GEM2_talos_training/**/gZ.txt')
    with open( '../GEM2_talos_training/gZ.txt', 'w' ) as result:
        for file_ in files:
            for line in open( file_, 'r' ):
                result.write( line )

  

    return 0

if __name__ == "__main__":
    main()