# Wine-Quality-Prediction

# Launch the Project
To run this project, you must Docker installed on your machine.

To build the image, you must be set on on the root of directory and execute the following command line : 

docker build -t wine .

To run a container of the image, use : 

docker run -d -p 8088:8000 -v "$(pwd)"/app/data:/data/ wine


The application can be accessed on your browser at the address : 
http://localhost:8088

And you can access to the virtual interface going to 

http://localhost:8088/docs