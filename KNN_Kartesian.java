package org.sparkexample;

import com.google.common.collect.Iterators;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.Vector;
import scala.Tuple3;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

public class KNN_Kartesian
{
    private static final Pattern SPACE = Pattern.compile(" ");
    static Vector parseVector(String line) {
        String[] splits = SPACE.split(line);
        double[] data = new double[splits.length];
        int i = 0;
        for (String s : splits) {
            data[i] = Double.parseDouble(s);
            i++;
        }
        splits = null;
        return new Vector(data);
    }

    public static class ReadData implements Function<String, Vector>{
        public Vector call(String s){
            return parseVector(s);
        }
    }



    public static void main (String[] args) throws FileNotFoundException, IOException{
        long startTime = System.currentTimeMillis();
        SparkConf sparkConf = new SparkConf().setAppName("TestKMeans");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        String trainFileName = args[0];
        String testFileName = args[1];
        final int dataDimension = Integer.parseInt(args[2]);
        final int numTestInstances = Integer.parseInt((args[3]));
        final int k = Integer.parseInt((args[4]));


        System.out.println("\n\n\n\nHello World from Spark!\n\n\n\n");
        JavaRDD<String> trainLines = sc.textFile(trainFileName);
        final JavaRDD<Vector> trainData = trainLines.map(new ReadData());
        Long trainDataCount = trainData.count();long currentTime1 = System.currentTimeMillis();
        System.out.println("\nTraining data read completed\nTime taken: "+(currentTime1-startTime)+"\n\n");


        JavaRDD<String> testLines = sc.textFile(testFileName);
        final JavaRDD<Vector> testData = testLines.map(new ReadData());
        Long testDataCount = testData.count();currentTime1 = System.currentTimeMillis();
        System.out.println("\nTest data read completed\nTime taken: "+(currentTime1-startTime)+"\n\n");

        JavaPairRDD<Vector, Long> trainZipped =  trainData.zipWithIndex();
        JavaPairRDD<Vector, Long> testZipped = testData.zipWithIndex();
        JavaPairRDD< Tuple2<Vector, Long>, Tuple2<Vector, Long> > zippedCart = trainZipped.cartesian(testZipped);
        //first tuple will be train, second tuple will be test
        Long cartCount = zippedCart.count();currentTime1 = System.currentTimeMillis();
        System.out.println("\nCartesian product completed\nTime taken: "+(currentTime1-startTime)+"\n\n");

        JavaPairRDD<Integer, Tuple2<Long, Double>> dPair = zippedCart.mapToPair(
                new PairFunction<Tuple2<Tuple2<Vector, Long>, Tuple2<Vector, Long>>, Integer, Tuple2<Long, Double>>() {
                    @Override
                    public Tuple2<Integer, Tuple2<Long, Double>> call(Tuple2<Tuple2<Vector, Long>, Tuple2<Vector, Long>> trainTestTuple2) throws Exception {
                        Tuple2 trainTuple = trainTestTuple2._1();
                        Tuple2 testTuple = trainTestTuple2._2();
                        Vector trainVector = (Vector) trainTuple._1();
                        Vector testVector = (Vector) testTuple._1();
                        double d = calculateDistance(trainVector, testVector);
                        Long testId = (Long) testTuple._2();
                        int testId_int = (int) (long) testId;
                        Long trainId = (Long) testTuple._2();
                        Tuple2<Long, Double> t2 = new Tuple2<Long, Double>(trainId, d);
                        return new Tuple2<Integer, Tuple2<Long, Double>>(testId_int, t2);
                    }


                    private double calculateDistance(Vector trainVector, Vector testVector) {
                        double d1 = 0;
                        for (int i = 0; i < dataDimension - 1; i++) {
                            d1 += Math.pow((trainVector.elements()[i] - testVector.elements()[i]), 2);
                        }
                        return Math.sqrt(d1);
                    }
                }
        );
        Long dPairDataCount = dPair.count();currentTime1 = System.currentTimeMillis();
        System.out.println("\nDistance pair generation completed\nTime taken: "+(currentTime1-startTime)+"\n\n");

        /*for(Tuple2<Integer, Tuple2<Long, Double>> tx: dPair.collect()){
            System.out.println(tx.toString());
        }*/

        JavaRDD<Long[]> nearestNeighborIndices = dPair.groupByKey().map(
                new Function<Tuple2<Integer, Iterable<Tuple2<Long, Double>>>, Long[]>() {
                    @Override
                    public Long[] call(Tuple2<Integer, Iterable<Tuple2<Long, Double>>> iterableT2) throws Exception {
                        Long[] nnIndices = new Long[k + 1];
                        ArrayList<Tuple2<Long, Double>> tempList = new ArrayList<Tuple2<Long, Double>>();
                        for (int i = 0; i < k; i++) {
                            tempList.add(new Tuple2<Long, Double>(1L, 999999.0));
                        }
                        for (Tuple2<Long, Double> t2 : iterableT2._2()) {
                            //if t2-distance is less than any of the item of tempList, then replace t2 with the max item of tempList
                            int checkIndex = checkNearestNeighbor(tempList, t2, k);
                            if (checkIndex != -1) {
                                tempList.set(checkIndex, t2);
                            }
                        }
                        for (int i = 0; i < k; i++) {
                            nnIndices[i] = tempList.get(i)._1();
                        }
                        nnIndices[k] = Long.valueOf(iterableT2._1());  // the last value of this array will hold the TestRow number
                        return nnIndices;
                    }

                    private int checkNearestNeighbor(ArrayList<Tuple2<Long, Double>> tempList, Tuple2<Long, Double> t2, int k) {
                        int flag = 0;
                        int ret = -1;  // if current instance doesn't belong to nearest neighbor List, return -1
                        // we will find the maximum distance value and it's index in the tempList
                        // if current t2->distance is less than any of the values in tempList, we will send the index of highest distance
                        double tempMaxValue = tempList.get(0)._2();
                        int tempMaxIndex = 0;
                        for (int i = 0; i < k; i++) {
                            if (t2._2() < tempList.get(i)._2()) {
                                flag = 1;
                            }
                            if (tempList.get(i)._2() > tempMaxValue) {      // check the max value and save index of maxValue
                                tempMaxIndex = i;
                                tempMaxValue = tempList.get(i)._2();
                            }
                        }
                        if (flag == 1) {
                            ret = tempMaxIndex;
                        }
                        return ret;
                    }
                }
        );
        Long nnIndicesCount = nearestNeighborIndices.count(); currentTime1 = System.currentTimeMillis();
        System.out.println("\nNearest neighbor inices calculation completed\nTime taken: "+(currentTime1-startTime)+"\n\n");


        /*for(Long[]l:nearestNeighborIndices.collect()){
            for (int x=0; x<k; x++){
                System.out.print(l[x] + " ");
            }
            System.out.println();

        }*/
        JavaRDD<Double> trainingClassLabels = trainData.map(
                new Function<Vector, Double>() {
                    public Double call(Vector vector) throws Exception {
                        int dataDimension = (int) vector.length();
                        return vector.elements()[dataDimension - 1];
                    }
                }
        );


        final List<Double>trainingClassLabelsList = trainingClassLabels.collect();
        JavaRDD<Integer[]> predictedLabels = nearestNeighborIndices.map(    //first element will be testRowNum, next is PredictedClassLabel
                new Function<Long[], Integer[]>() {
                    @Override
                    public Integer[] call(Long[] indices) throws Exception {
                        Integer[] ret = new Integer[2];
                        int testRowNum = indices.length;
                        ret[0] = (int) (long) indices[testRowNum - 1];
                        ret[1] = predictClassLabels(indices, trainingClassLabelsList);
                        return ret;
                    }

                    private int predictClassLabels(Long[] indices, List<Double> trainingClassLabelsList) {
                        int zeroSum = 0;
                        int oneSum = 0;
                        for (int i = 0; i < indices.length - 1; i++) {            // the last element of indices array is the testRowNum
                            int currentIndex = (int) (long) indices[i];
                            if (trainingClassLabelsList.get(currentIndex) == 0) {
                                zeroSum++;
                            } else {
                                oneSum++;
                            }
                        }
                        if (oneSum >= zeroSum) {
                            return 1;
                        } else {
                            return 0;
                        }
                    }
                }
        );
        Long lablesCount = trainingClassLabels.count(); currentTime1 = System.currentTimeMillis();
        System.out.println("\nclass label generation completed\nTime taken: "+(currentTime1-startTime)+"\n\n");


        Random randomGenerator = new Random();
        int randomInt = randomGenerator.nextInt(100000);
        String outFileName = "labels_"+Integer.toString(k)+"_"+Integer.toString(dataDimension)+"_"+Integer.toString(randomInt)+".txt";
        predictedLabels.saveAsTextFile(outFileName);
        System.out.println("Predicted Labels output written to " + outFileName + "\n\n");

/*        System.out.println("\n\n\n\n");for(Integer[]a : predictedLabels.collect()){
            System.out.println(a[0]+" "+ a[1]);
        }System.out.println("\n\n\n\n");*/

        sc.stop();
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("\n\n\nEnd of Spark KNN CLassification\nTotal time taken: " + totalTime + "\n\n\n");

    }
}
