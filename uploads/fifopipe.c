#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/types.h>
#include<string.h>
#include<stdlib.h>

#define MAX 1024

void main()
{
    int pipe1[2],pipe2[2];
    char pipe1msg[20]="pipe1";
    char pipe2msg[20]="pipe2";
    char inbuf[20];
    int pid,return1,return2;
    return1=pipe(pipe1);
    if(return1==-1){
        printf("\n unable to open pipe1");
        exit(0);
    }
    return2=pipe(pipe2);
    if(return2==-1)
    {
        printf("\n unable to open pipe2");
        exit(0); 
    }
    pid=fork();
    if(pid>0)                      //parent process
    {
        close(pipe1[0]);
        close(pipe2[1]);
        printf("\n parent writing to pipe1");
        write(pipe1[1],pipe1msg,strlen(pipe1msg));
        read(pipe2[0],inbuf,sizeof(inbuf));
        printf("\n inbuf:%s",inbuf);
    }


}