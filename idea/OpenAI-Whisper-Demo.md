# OpeanAI Whisper Demo
If you have docker, try this:

``` sh
cat <<EOF > /tmp/docker-init.sh
apt update && apt install python3-pip ffmpeg git -y
git clone https://gist.github.com/kpe/6a70395ce171ffee43d927eaf90b81b6 /tmp/whisper
cd /tmp/whisper
pip3 install -r requirements.txt
python3 -m whisper_demo
EOF
docker run -ti --rm --name whisper -p 7860:7860/tcp -v /tmp/docker-init.sh:/tmp/init.sh ubuntu /bin/bash --rcfile /tmp/init.sh
```
(this would download quite some stuff, so you might consider skipping the `--rm` option above)

or if you have an Ubuntu VM, you could try something along these lines:

``` sh
sudo apt update && sudo apt install python3-pip3 ffmpeg git -y
git clone https://gist.github.com/kpe/6a70395ce171ffee43d927eaf90b81b6 /tmp/whisper
cd /tmp/whisper
pip3 install -r requirements.txt
python3 -m whisper_demo
```

And at last click on the [https://<uid>-gradio.live](#) link shown in the console
