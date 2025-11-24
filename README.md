minikube start
kubectl create deployment mynginx --image=nginx
if already created then 
kubectl set image deployment/myngnix nginx=nginx:latest
kubectl get deployments
kubectl get pods
kubectl describe pods
kubectl expose deployment mynginx --type=NodePort --port=80 --target-port=80
kubectl scale deployment mynginx --replicas=4
kubectl get svc
kubectl get service mynginx
kubectl port-forward svc/mynginx 8081:80
minikube tunnel
minikube service mynginx --url
Minikube dashboard

for stop and clean

kubectl delete deployment mynginx
kubectl delete service mynginx
minikube stop
minikube delete


Nagios Automation Steps

docker pull jasonrivers/nagios:latest
docker run --name nagiosdemo -p 8888:80 jasonrivers/nagios:latest
Open your browser and navigate to:http://localhost:8888
Login Credentials:
Username: nagiosadmin
Password: nagios
docker stop nagiosdemo
docker rm nagiosdemo
docker images
docker rmijasonrivers/nagios:latest

------------------------------
open aws acedamy
press green button(Aws)
open EC2->Launch instance
name:MyExampleWebServer 
ubuntu
instance type-t2.micro
create key pair
->MyExampleKeyPair
->RSA
->pem
Allow hhtps
Allow http
launch instance
connect to instance
ssh client link copy
public ip
connect
create a empty floder(pem file should be in this floder(WEB))
OPEN POWERSHELL
copy ssh client link copy(Here copy the ssh – i command from SSH client connect tab)
sudo apt update
sudo apt-get install docker.io
sudo apt install git
Sudo apt install nano
:-  git clone <paste the github link of maven-web-java project>(project repo with index.html)
FROM THE PROJECT FLODER(WEB) OPEN GIT BASH
nano Dockerfile 
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
ctrl+s
ctrl+x
sudo docker build -t mywebapp . (or) docker build -t mywebapp . 
sudo docker run -d -p 80:80 mywebapp (or) docker run -d -p 80:80 mywebapp
OPEN DOCKER AND OPEN WITH IP ADDRESS AND YOU GET HELLO WORLD.
