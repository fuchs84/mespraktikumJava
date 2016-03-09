import Classify.Test;
import WekaAlgorithm.Controller;
import WekaAlgorithm.VariationConstants;

import java.util.ArrayList;


/**
 * Main starts the program
 */
public class Main {
    private static Controller controller;
    private static VariationConstants variationConstants;

    /**
     * Main-method
     * @param args passing arguments
     */
    public static void main(String[] args) {
        long start = System.currentTimeMillis();

        String patternPathAll = "";
        String labelPathAll = "";
        String patternPathPass = "";
        String labelPathPass = "";
        controller = new Controller();
        controller.init(patternPathAll, labelPathAll, patternPathPass, labelPathPass);
        controller.evaluation("crossValidation");

        System.out.println("Time: " + (System.currentTimeMillis() - start));
    }
}
