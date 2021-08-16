# https://github.com/1Konny/FactorVAE/blob/master/scripts/prepare_data.sh
# https://github.com/yunjey/stargan/blob/master/download.sh

if [ "$1" = "3DChairs" ]; then
    mkdir -p data
    cd data
    echo "download 3DChairs dataset."
    wget https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar

    if [ -f "rendered_chairs.tar" ]; then
        tar -xvf rendered_chairs.tar
        root=rendered_chairs
        new_root="3DChairs/images"
        rm $root"/all_chair_names.mat"
        mkdir -p $new_root
        n=1
        for dir in `ls -1t $root`; do
            for imgpath in `ls -1t $root/$dir/renders/*`; do
                imgname=$(echo "$imgpath" | cut -d"/" -f4)
                newpath=$img" "$new_root"/"$n"_"$imgname
                mv $imgpath $newpath
                n=$((n+1))
            done
        done
        rm -rf $root

    else
        echo "Download uncompleted"
    fi

elif [ "$1" = "dsprites" ]; then
    mkdir -p data
    cd data
    git clone https://github.com/deepmind/dsprites-dataset.git
    cd dsprites-dataset
    rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
    cd ../dataloader
    python preprocess_dsprites.py

elif [ "$1" = "CelebA" ]; then
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE
fi

