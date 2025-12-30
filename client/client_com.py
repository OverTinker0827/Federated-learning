import socket
import struct
import torch
import io
class Client_Com:
    def __init__(self,server_ip,port):
        self.client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            self.client.connect((server_ip,port))
        except Exception as e:
            print("exception occured")


    def send(self,state_dict):
        buffer=io.BytesIO()
        torch.save(state_dict,buffer)
        weights_in_bytes=buffer.getvalue()
        try:
            self.client.sendall(struct.pack("!Q",len(weights_in_bytes)))

            self.client.sendall(weights_in_bytes)  
            return True
        except Exception as e:
            return False      
    def recv_with_size(self,n):
        data=b""
        while len(data)<n:
            chunk=self.client.recv(n-len(data))
            if not chunk:
                raise ConnectionError("data stopped arriving to client")
            data+=chunk
        return data
    def recieve_id(self):
        msg=self.recv_with_size(4)
        client_id=struct.unpack("!I", msg)[0]
        return client_id
    def recieve_weights(self):
        recv=struct.unpack("!Q",self.recv_with_size(8))
        print(recv)
        size=recv[0]
        weight_in_bytes=self.recv_with_size(size)
        buffer=io.BytesIO(weight_in_bytes)
        state_dict=torch.load(buffer,map_location="cpu")
        return state_dict
