using System;
using System.Collections.Generic;
using NeatNetwork.Libraries;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            NN n = new NN(new int[] { 4, 5, 6, 5, 4, 3, 1 }, Activation.ActivationFunctions.Sigmoid, 1.5, -1.5, .5);
            List<double[]> X = new List<double[]>()
            {
                new double[] { 3, 7, 8, 9 },
            };

            double targetVal = 0.63541524;
            List<double[]> y = new List<double[]>()
            {
                new double[] { targetVal },
            };

            double learningRate = .1;
            for (int i = 0; i < 4000000; i++)
            {
                n.SupervisedLearningBatch(X, y, 1, Cost.CostFunctions.SquaredMean, learningRate);
                double output;
                Console.WriteLine(output = n.Execute(X[0])[0]);
                if (targetVal == output)
                {
                    Console.WriteLine($"Overfitted in {i} steps");
                    Console.ReadKey();
                    break;
                }
            }

            
        }
    }
}
