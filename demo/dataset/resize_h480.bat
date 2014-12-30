mkdir out
FOR %%A IN (*.jpg) DO (
    c:\tools\ffmpeg\bin\ffmpeg -i "%%A" -vf "scale=-1:480, transpose=1" "out/%%A"
)