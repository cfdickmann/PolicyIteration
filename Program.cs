using System;
using System.Linq;
using System.Collections.Generic;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Math;

namespace ConsoleApp1
{
    class Program
    {
        static double r = 0.03;
        static double div = 0.04;
        static int n = 5;
        static double strike = 0.9;
        static double sigma = 0.3;

        static MathNet.Numerics.Distributions.Normal nd = new MathNet.Numerics.Distributions.Normal(0, 1);

        public static double[] path(double spot, double drift, double vola)
        {
            double dt = 1.0 / (n - 1);

            var pa = new double[n];

            pa[0] = spot;

            for (int i = 1; i < n; i++)
            {
                pa[i] = pa[i - 1] * Math.Exp((drift - 0.5 * vola * vola) * dt + Math.Sqrt(dt) * vola * nd.Sample());
            }

            return pa;
        }

        public static double[] payoff(double[] p)
        {
            var pa = new double[n];
            for (int i = 0; i < n; i++)
            {
                pa[i] = Math.Max(strike - p[i], 0.0) * Math.Exp(-r * (double)i * 1.0 / (double)n);
            }
            return pa;
        }

        public static List<double[]> allPaths(int number)
        {
            var ran = new Random();

            var li = new List<double[]>();

            for (int i = 0; i < number; i++)
            {
                double spot = 1;// ran.NextDouble() * 2.0;
                double drift = r - div;// ran.NextDouble() * 2.0 - 1.0;
                double vola = sigma;// ran.NextDouble() * 1.0;
                li.Add(path(spot, drift, vola));
            }

            return li;
        }

        public static double readNet(Accord.Neuro.ActivationNetwork net, double spot, double strike, double r, double div, double time, double vola)
        {
            if (net == null) return 1000000;

            double factorA = spot;
            double factorB = 1.0 / time;

            double strikeBar = strike / factorA;
            double driftBar = (r - div) * factorB;
            double volaBar = sigma * Math.Sqrt(factorB);

            return factorA * net.Compute(new double[] { strikeBar, driftBar, volaBar })[0];
        }

        public static double[] contins(Accord.Neuro.ActivationNetwork svm, double[] p)
        {
            double[] re = new double[p.Length];

            for (int k = 0; k < p.Length; k++)
            {
                double continuationValue = readNet(svm, p[k], strike, r, div, (double)(n - k) / (double)(n), sigma);
             //   Console.WriteLine("cont: " + continuationValue);
            }

            return re;
        }

        public static int ausw(double[] a, double[] b, int start)
        {
            for (int i = start; i < a.Length - 1; i++)
            {
                if (a[i] > b[i]) return i;
            }
            return a.Length - 1;
        }

        public static List<Tuple<double[], double>> addexamples(Accord.Neuro.ActivationNetwork svm, double[] path)
        {
            var lis = new List<Tuple<double[], double>>();

            double[] ps = payoff(path);
            double[] conts = contins(svm, path);

            for (int k = 0; k < n; k++)
            {
                int iEx = ausw(ps, conts, k + 1);
                double cont_real = ps[iEx];

                double ret = cont_real > ps[k] ? cont_real : ps[k];

                double time = 1.0 - (double)(k) / (double)(n);
                double[] data = new double[] { strike / path[k], (r - div) / time, sigma * Math.Sqrt(time) };

                lis.Add(new Tuple<double[], double>(data, ret / path[k]));
            }

            return lis;
        }

        public static List<Tuple<double[], double>> getStoppedPayoffs(Accord.Neuro.ActivationNetwork svm, List<double[]> paths)
        {
            var lis = new List<Tuple<double[], double>>();
            
            for (int i = 0; i < paths.Count; i++)
            {
                lis = lis.Concat(addexamples(svm, paths[i])).ToList();
            }

            return lis;
        }

        //public static void test1()
        //{
        //    var pa = allPaths(100000);
        //    List<double[]> ds = pa.Select(item => payoff(item)).ToList();
        //    double mean = ds.Select(i => i[n - 1]).ToArray().Average();
        //    Console.WriteLine("Mean: " + mean);
        //    if (Math.Abs(mean - 0.13) < 0.01) Console.WriteLine("test1 passed.");
        //}

        public static void test2()
        {
            var li = new List<Tuple<double[], double>>();
            Random r = new Random();

            for (int i = 0; i < 10000; i++)
            {
                double a = r.NextDouble();
                double b = r.NextDouble();
                double c = r.NextDouble();

                li.Add(new Tuple<double[], double>(new double[] { a, b, c }, a * b + c));
            }

            var net = trainNet(li);
            double output = net.Compute(new double[] { 0.5, 0.1, 0.9 })[0];

            Console.WriteLine("output: " + output);
            if (Math.Abs(output - 0.95) < 0.01) Console.WriteLine("test2 passed.");
        }


        public static void printliste( List<Tuple<double[], double>> li)
        {
            Console.WriteLine("Liste: ");
            for (int i = 0; i < li.Count; i++)
            {
                var a = li[i];
                for (int k = 0; k < a.Item1.Length; k++)
                {
                    Console.Write(a.Item1[k] + " ,");
                }

                Console.Write( a.Item2);
            }

        }

        [MTAThread]
        static void Main(string[] args)
        {

            test2();

            int N = 3;
            var paths = allPaths(3000);

            ActivationNetwork network = null;

            for (int i = 0; i < N; ++i)
            {
                var liste = getStoppedPayoffs(network, paths);

               // printliste(liste);
                network = trainNet(liste, verbose: true);
            }

            Console.ReadKey();
        }

        public static ActivationNetwork trainNet(List<Tuple<double[], double>> inp, bool verbose = false)
        {
            double[][] input = inp.Select(item => item.Item1).ToArray();
            double[][] output = inp.Select(item => new double[] { item.Item2 }).ToArray();

            double learningrate = 0.1;
            int iterations = 100;
            double sigmoidAlphaValue = 0.5;
            int neuronsInFirstLayer = 1;
            bool useNguyenWidrow = true;
            bool useRegularization = false;

            // create multi-layer neural network
            ActivationNetwork network = new ActivationNetwork(new BipolarSigmoidFunction(sigmoidAlphaValue), 3, neuronsInFirstLayer, 1);
            if (useNguyenWidrow)
                new NguyenWidrow(network).Randomize();
            var teacher = new LevenbergMarquardtLearning(network, useRegularization);
            teacher.LearningRate = learningrate;

            for (int i = 0; i < iterations; ++i)
            {
                double error = teacher.RunEpoch(input, output) / input.Length;

                if (verbose)
                    Console.WriteLine("Iteration " + i + ", error:" + error);
            }

            return network;
        }
    }
}

