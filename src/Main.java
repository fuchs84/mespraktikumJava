import Users.Matthias;
import Users.Sebastian;

/**
 * Created by MatthiasFuchs on 06.11.15.
 */
public class Main {
    private static Matthias matthias;
    private static Sebastian sebastian;
    public static void main(String[] args)  {
        sebastian = new Sebastian();
        matthias = new Matthias();
        sebastian.run();
        //matthias.run();

    }
}
