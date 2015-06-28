/*

  Neuro.NET - Library of neural networks for .NET
  Copyright (C) 2001-2015  Alex Cherkasov. All rights reserved.
                           email: info@xpidea.com
                           web: http://xpidea.com/

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
  02110-1301, USA.

{   CREDITS:                                                    }
{   This work is based on publications of:                      }
{          -Christopher M. Bishop                               }
{          -Jose C. Principe                                    }
{          -Samuel J. Rogers                                    }
{          -Laurene V. Fausett                                  }
{          -Simon S. Haykin                                     }
  
*/

using System;
using xpidea.neuro.net.backprop;
using xpidea.neuro.net.patterns;

namespace xpidea.neuro.net.examples.backprop.xor
{
    internal class Class1
    {
        private const int TrainingSets = 4;

        private static void SetPattern(Pattern aPattern, double x, double y, double z)
        {
            aPattern.Input[0] = x;
            aPattern.Input[1] = y;
            aPattern.Output[0] = z;
        }

        [STAThread]
        private static void Main(string[] args)
        {
            Console.Out.WriteLine("                       BACKPROPAGATION neural network demo.");
            Console.Out.WriteLine("");
            Console.Out.WriteLine("												 Copyright(C) XP Idea.com 2001-2004 ");
            Console.Out.WriteLine("");
            Console.Out.WriteLine("The purpose of this demo is to show learning abilities of BACKPROP network.");
            Console.Out.WriteLine("The BACKPROP network is able to learn much more complex data patterns, than");
            Console.Out.WriteLine("Adaline network (please see OCR demo application). ");
            Console.Out.WriteLine("This example simple shows that the Backprop network is able to learn ");
            Console.Out.WriteLine("an 'exclusive OR' (XOR) operation, but the Adaline network is not able to do so.");
            Console.Out.WriteLine("");
            Console.Out.WriteLine("         false XOR false = false");
            Console.Out.WriteLine("         true XOR false = true");
            Console.Out.WriteLine("         false XOR true = true");
            Console.Out.WriteLine("         true XOR true = false");
            Console.Out.WriteLine("");
            Console.Out.WriteLine(" As result of the training, the network will produce output ‘0’");
            Console.Out.WriteLine("corresponding to logical ‘false’ or ‘1’ corresponding to logical ‘true’ value.");
            Console.Out.WriteLine("");
            Console.Out.WriteLine("PLEASE HIT ENTER TO CONTINUE");
            Console.In.ReadLine();

            Console.Out.WriteLine("");
            Console.Out.WriteLine("During this demo you will be prompted to enter input values");
            Console.Out.WriteLine("for the network. Then network will perform “XOR” operation on ");
            Console.Out.WriteLine("the entered values and result will be displayed to you. ");
            Console.Out.WriteLine("Please enter any values in range from 0 to 1 and hit [ENTER] when prompted. ");
            Console.Out.WriteLine("");
            Console.Out.WriteLine("NOW THE NETWORK IS READY TO LEARN FOLLOWING PATTERNS");
            Console.Out.WriteLine("");
            Console.Out.WriteLine("			false XOR false = false;");
            Console.Out.WriteLine("			true XOR false = true;");
            Console.Out.WriteLine("			false XOR true = true;");
            Console.Out.WriteLine("			true XOR true = false;");
            Console.Out.WriteLine("PLEASE HIT ENTER TO BEGIN TRAINING");
            Console.In.ReadLine();
            Console.Out.Write("TRAINING....");

            double d;
            BackPropagationNetwork BackPropNet;
            var patterns = new PatternsCollection(TrainingSets, 2, 1);

            SetPattern(patterns[0], 0, 0, 0);
            SetPattern(patterns[1], 0, 1, 1);
            SetPattern(patterns[2], 1, 0, 1);
            SetPattern(patterns[3], 1, 1, 0);
            //Network(0.55,0.6,
            BackPropNet = new BackPropagationNetwork(0.55, 0.6, new int[3] {2, 3, 1});
            BackPropNet.Train(patterns);
            Console.Out.WriteLine("DONE!");
            Console.Out.WriteLine("");
            //BackPropNet.SaveToFile("test.net");
            while (true)
            {
                try
                {
                    Console.Out.Write("Enter 1st value: ");
                    d = double.Parse(Console.In.ReadLine());
                    BackPropNet.InputNode(0).Value = d;
                    Console.Out.Write("Enter 2nd value: ");
                    d = double.Parse(Console.In.ReadLine());
                    BackPropNet.InputNode(1).Value = d;
                    BackPropNet.Run();
                    Console.Out.WriteLine("Result: " + Math.Round(BackPropNet.OutputNode(0).Value));
                    Console.Out.WriteLine("");
                }
                catch
                {
                    return;
                }
            }
        }
    }
}