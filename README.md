## Build and Run

Use [Maven](https://maven.apache.org/) to build the examples. 

```
mvn clean package
```
```
(1) mvn install

(2) java -cp dl4j-examples/target/dl4j-examples-0.4-rc0-SNAPSHOT-bin.jar "org.deeplearning4j.examples.dataExamples.CSVExample" XOR.csv 2 6 250 .1 1000
```
## Training on custom .csv (comma separated) file data
(1) Put your csv data files in [dl4j-examples/src/main/resources](https://github.com/jsa41394/DeepLearning4Java-CSV-Data-Eg/tree/master/dl4j-examples/src/main/resources)

(2) Note the number of rows, cols, etc. in your csv file

(3) Build and run with command line arguments for any data file

Note: must rebuild when adding new data files or modifying [CSVExample.java](https://github.com/jsa41394/DeepLearning4Java-CSV-Data-Eg/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/CSVExample.java)



## Original Documentation (DeepLearning4Java)
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

