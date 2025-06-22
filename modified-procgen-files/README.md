In order to add the coinrun-left environment to procgen,
follow the following instructions,
which will patch the cloned code with the necessary changes

    git clone https://github.com/openai/procgen
    cd procgen
    git checkout 5e1dbf3
    patch -p1 < ../modified-procgen-files/procgen.diff
