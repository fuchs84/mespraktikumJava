package WekaAlgorithm;

import WekaAlgorithm.Classifier.AbstractClassifier;
import WekaAlgorithm.Classifier.ClassifierJ48;
import WekaAlgorithm.Classifier.ClassifierNB;
import WekaAlgorithm.Classifier.ClassifierRF;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */
public class Controller {
    private DataGenerator dataGenerator;
    private ArrayList<AbstractClassifier> classifiers;
    private int numberOfClassifiers;
    private boolean built = false;
    private String[] selectedClassifiers = {"NaiveBayes", "NaiveBayes", "DecisionTree", "DecisionTree", "DecisionTree"
            , "DecisionTree", "DecisionTree", "RandomForest", "DecisionTree", "DecisionTree", "DecisionTree"};

    Instances[] allInstances;
    Instances[] passInstances;

    public Controller() {
        dataGenerator = new DataGenerator();
    }

    public void init(String patternPathAll, String labelPathAll, String patternPathPass, String labelPathPass) {
        try {
            passInstances = dataGenerator.buildTrain(patternPathPass, labelPathPass);
            allInstances = dataGenerator.buildTrain(patternPathAll, labelPathAll);
            buildStructure();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void buildStructure() throws Exception {
        numberOfClassifiers = allInstances.length + passInstances.length;
        classifiers = new ArrayList<>();
        if(selectedClassifiers.length == numberOfClassifiers) {
            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier;
                if(selectedClassifiers[i].equals("DecisionTree")) {
                    System.out.println("DecisionTree");
                    classifier = new ClassifierJ48();
                    classifiers.add(classifier);
                } else if(selectedClassifiers[i].equals("NaiveBayes")) {
                    System.out.println("NaiveBayes");
                    classifier = new ClassifierNB();
                    classifiers.add(classifier);
                } else if(selectedClassifiers[i].equals("RandomForest")) {
                    System.out.println("RandomForest");
                    classifier = new ClassifierRF();
                    classifiers.add(classifier);
                }
            }
            built = false;
        } else {
            System.out.println("numberOfClassifiers is not equal to selectedClassifiers");
        }
    }

    public void train() {
        try {
            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier = new ClassifierJ48();
                if(i < passInstances.length) {
                    classifier.setInstances(passInstances[i]);
                } else {
                    classifier.setInstances(allInstances[i - passInstances.length]);
                }
                classifier.train();
                classifiers.add(classifier);
            }
            built = true;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void classify(String patternPath) {
        double[][] labels = new double[numberOfClassifiers][];
        try {
            Instances instances = dataGenerator.buildClassify(patternPath);
            if(built == false) {
                train();
            }
            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier = classifiers.get(i);
                classifier.setInstances(instances);
                labels[i] = classifier.classify();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void evaluation(String type) {
        try {
            FileWriter fw = new FileWriter("evaluation.txt");

            ClassifierEvaluation eval = new ClassifierEvaluation();
            if(type.equals("crossValidation")) {
                for(int i = 0; i < numberOfClassifiers; i++) {
                    AbstractClassifier classifier = classifiers.get(i);
                    fw.append(selectedClassifiers[i] + "\n");
                    if(i < passInstances.length) {
                        eval.crossValidation(classifier.getClassifier(), 10, passInstances[i], fw);
                    } else {
                        eval.crossValidation(classifier.getClassifier(), 10, allInstances[i - passInstances.length], fw);
                    }
                    fw.append("\n"+ "\n");
                }
            } else if(type.equals("percentageSplit")) {
                for(int i = 0; i < numberOfClassifiers; i++) {
                    AbstractClassifier classifier = classifiers.get(i);
                    if(i < passInstances.length) {
                        eval.percentageSplit(classifier.getClassifier(), passInstances[i], fw);
                    } else {
                        eval.percentageSplit(classifier.getClassifier(), allInstances[i - passInstances.length], fw);
                    }
                    fw.append("\n"+ "\n");
                }
            }
            fw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
