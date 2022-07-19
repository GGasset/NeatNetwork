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
            // Supervised learning demonstration
            /*NN n = new NN(new int[] { 4, 5, 6, 5, 4, 3, 1 }, Activation.ActivationFunctions.Sigmoid, 1.5, -1.5, .5);
            List<double[]> X = new List<double[]>()
            {
                new double[] { 3, 7, 8, 9 },
            };

            double targetVal = 0.63541524;
            List<double[]> y = new List<double[]>()
            {
                new double[] { targetVal },
            };

            double learningRate = .75;
            for (int j = 0; j < 4000; j++)
            {
                n.SupervisedLearningBatch(X, y, 1, Cost.CostFunctions.SquaredMean, learningRate);
                double output;
                Console.WriteLine(output = n.Execute(X[0])[0]);
                if (targetVal == output)
                {
                    Console.WriteLine($"Overfitted in {j} steps");
                    Console.ReadKey();
                    break;
                }
            }

            n = new NN(n.ToString());

            Console.WriteLine($"{n.Execute(X[0])[0]}");
            Console.ReadKey();*/
            
            // Reinforcement learning demonstration
            /*double learningRate = 1;
            ReinforcementLearningNN agent = new ReinforcementLearningNN(new NN(new int[] { 1, 15, 4, 400, 1 }, Activation.ActivationFunctions.Sigmoid), learningRate);
            double[] input = { 30 };
            double biggestOutput = 0;
            for (int i = 0; i < 5000; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    double output = agent.Execute(input)[0];
                    Console.WriteLine(output);
                    if (i != 0)
                    {
                        agent.GiveReward(output - biggestOutput);
                        biggestOutput += (output - biggestOutput) * Convert.ToInt32(output > biggestOutput);
                    }
                    else
                        biggestOutput = output;
                }
                agent.TerminateAgent();
            }*/
        }
    }
}
