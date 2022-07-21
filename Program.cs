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
             List<double[]> X = new List<double[]>()
             {
                 new double[] { 0,0 },
                 new double[] { 1,0 },
                 new double[] { 0,1 },
                 new double[] { 1,1 },
             };

             List<double[]> y = new List<double[]>()
             {
                 new double[] { 1 },
                 new double[] { -1 },
                 new double[] { -1 },
                 new double[] { 1 },
             };


            // Evolution Learning demonstration
            NNEvolutionManager world = new NNEvolutionManager(15000, new int[] { 2, 1 }, Activation.ActivationFunctions.Sigmoid, 1.5, -1.5, 0.37, 1, .25);


            for (int i = 0; i < 4000; i++)
            {
                while (!world.AreAllNetworksScored())
                {
                    NN cn = world.GetNextToScoreNetwork();
                    double meanCost = 0;
                    for (int j = 0; j < X.Count; j++)
                    {
                        meanCost += Cost.GetCost(cn.Execute(X[j]), y[j], Cost.CostFunctions.SquaredMean);
                    }
                    meanCost /= X.Count;
                    world.SetNextNetworkToBeScoredScore(-meanCost);

                    /*double score = cn.Execute(X[1])[0];
                    world.SetNextNetworkToBeScoredScore(score);*/
                }
                Console.WriteLine($"Generation: {i}   NetworkCount: {world.Networks.Count}   MaxScore: {world.MaxScore}");

                world.HaveChild(20, 50);
            }
            Console.WriteLine("\n");
            NN n = world.GetMaxScoredNetwork();
            for (int i = 0; i < X.Count; i++)
            {
                double output = n.Execute(X[i])[0];
                double expected = y[i][0];
                Console.WriteLine($"Output: {output}, Expected: {expected}");
            }

            // Supervised learning demonstration
            /*NN world = new NN(new int[] { 4, 5, 6, 5, 4, 3, 1 }, Activation.ActivationFunctions.Sigmoid, 1.5, -1.5, .5);
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
                world.SupervisedLearningBatch(X, y, 1, Cost.CostFunctions.SquaredMean, learningRate);
                double output;
                Console.WriteLine(output = world.Execute(X[0])[0]);
                if (targetVal == output)
                {
                    Console.WriteLine($"Overfitted in {j} steps");
                    Console.ReadKey();
                    break;
                }
            }

            world = new NN(world.ToString());

            Console.WriteLine($"{world.Execute(X[0])[0]}");
            Console.ReadKey();*/
            
            // Reinforcement Learning demonstration
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
