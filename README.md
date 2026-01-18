# Federated-learning

RUN dashboard using 
```
npm start
```

Configure dashboard ip to public ip in .env file

RUN each client using 
```bash
python3 .clientx/run_api.py
```
 in its directory.

RUN server api endpoints using
 ```bash
python3 ./server/run_api.py
```

 Change the client ips in dashboard/src/client_config.json with the private ips.
 
 Change the server ip in client_/config.py to its private ip

