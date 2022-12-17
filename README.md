# Wine-Quality-Prediction

# Contributors

Matthieu Cabrera
Laura Trujillo

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


# Test the project
 
To test the projet, stop the docker and execute the following command line from the /app folder : 

pytest --cov=. tests/


# Files

A raw description of the files can be found at the beginning of each of them. Furthermore, you can refer to the documentation. 

Note that app/model/.exploration.py is a file that has been used to explore the dasatet and try 4 different models that turned out to be equivalent in terms of test accuracy. This file is not part of the application but was useful to create it. Same thing with best_wine_methode 1 and 2.

