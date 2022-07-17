using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            NN n = new NN(new int[] { 4, 5, 6, 5, 4, 3, 1 }, Libraries.Activation.ActivationFunctions.Sigmoid, 1.5, -1.5, 0);
            List<double[]> X = new List<double[]>()
            {
                new double[] { 3, 7, 8, 9 },
            };

            List<double[]> y = new List<double[]>()
            {
                new double[] { 1 },
            };

            for (int i = 0; i < 4000000; i++)
            {
                n.SupervisedLearningBatch(X, y, 1, Libraries.Cost.CostFunctions.SquaredMean);
                Console.WriteLine(n.Execute(X[0])[0]);
            }

        }
    }
}
