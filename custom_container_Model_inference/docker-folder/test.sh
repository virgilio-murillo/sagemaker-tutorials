docker rmi -f $(docker images -q)
docker rm $(docker ps -aq)
docker build . -t tmp
docker run -p 8081:8080 tmp
