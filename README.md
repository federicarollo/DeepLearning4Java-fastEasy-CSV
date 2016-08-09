DL4J Release 0.4 Examples 
=========================
Repository of Deeplearning4J neural net examples:

- MLP Neural Nets
- Convolutional Neural Nets
- Recurrent Neural Nets
- TSNE
- Word2Vec & GloVe
- Anomaly Detection

---

## Build and Run

Use [Maven](https://maven.apache.org/) to build the examples. 

```
mvn clean package
```

Run the `runexamples.sh` script to run the examples (requires [bash](https://www.gnu.org/software/bash/)). It will list the examples and prompt you for the one to run. Pass the `--all` argument to run all of them. (Other options are shown with `-h`).

```
./runexamples.sh [-h | --help]

if error: ... $'\r': command not found
run: sed -i 's/\r$//' <file>

Test a single example:
java -cp dl4j-examples/target/dl4j-examples-0.4-rc0-SNAPSHOT-bin.jar "org.deeplearning4j.examples.feedforward.xor.XorExample"

or 
java -cp dl4j-examples/target/dl4j-examples-0.4-rc0-SNAPSHOT-bin.jar "org.deeplearning4j.examples.convolution.LenetMnistExample"

or
java -cp dl4j-examples/target/dl4j-examples-0.4-rc0-SNAPSHOT-bin.jar "org.deeplearning4j.examples.dataExamples.CSVExample"
```


## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.


