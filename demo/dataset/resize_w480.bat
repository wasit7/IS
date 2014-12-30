mkdir out
FOR %%A IN (*.jpg) DO (
    c:\tools\ffmpeg\bin\ffmpeg -i "%%A" -vf "scale=480:-1" "out/%%A"
)