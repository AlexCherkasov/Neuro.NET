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
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using xpidea.neuro.net.patterns;
using xpidea.neuro.net.son;

namespace xpidea.neuro.net.examples.son.movie
{
    /// <summary>
    ///     Summary description for Form1.
    /// </summary>
    public class PatternStyles
    {
        public static bool IsTerminated;

        public static void BuildSquare(out double x, out double y)
        {
            x = NeuroObject.Random(-1, 1);
            y = NeuroObject.Random(-1, 1);
        }

        public static void BuildEvenSquare(PatternsCollection aData)
        {
            var c = (int) Math.Round(Math.Sqrt(aData.Count));
            var s = 2/(double) c;
            double x = -1;
            double y = -1;
            for (var i = 0; i < c; i++)
            {
                for (var j = 0; j < c; j++)
                {
                    aData[i*c + j].Input[0] = x;
                    aData[i*c + j].Input[1] = y;
                    y = y + s;
                }
                y = -1;
                x = x + s;
            }
        }

        public static void BuildEvenCircle(PatternsCollection aData)
        {
            BuildEvenSquare(aData);
            for (var i = aData.Count - 1; i >= 0; i--)
            {
                var x = aData[i].Input[0];
                var y = aData[i].Input[1];
                if ((x*x + y*y) > 0.8)
                    aData.RemoveAt(i);
            }
        }

        public static void BuildDimond(out double x, out double y)
        {
            do
            {
                x = NeuroObject.Random(-1, 1);
                y = NeuroObject.Random(-1, 1);
            } while (!((Math.Abs(x) + Math.Abs(y)) <= 1));
        }

        public static void ChessBoard(out double x, out double y)
        {
            do
            {
                x = NeuroObject.Random(-1, 1);
                y = NeuroObject.Random(-1, 1);
            } while (Math.Sign(x) != -Math.Sign(y));
        }

        public static void BuildPlus(out double x, out double y)
        {
            do
            {
                x = NeuroObject.Random(-1, 1);
                y = NeuroObject.Random(-1, 1);
            } while (!((Math.Abs(x) < 0.2) || (Math.Abs(y) < 0.2)));
        }

        public static void BuildRing(out double x, out double y)
        {
            do
            {
                x = NeuroObject.Random(-1, 1);
                y = NeuroObject.Random(-1, 1);
            } while (!(((x*x + y*y) < 0.5) && ((x*x + y*y) > 0.3)));
        }

        public static void BuildCircle(out double x, out double y)
        {
            do
            {
                x = NeuroObject.Random(-1, 1);
                y = NeuroObject.Random(-1, 1);
            } while (!((x*x + y*y) < 0.4));
        }

        public static void BuildTwoCircles(out double x, out double y)
        {
            var c = 0.3;
            var r = 0.1;
            do
            {
                x = NeuroObject.Random(-1, 1);
                y = NeuroObject.Random(-1, 1);
            } while (!((((x - c)*(x - c) + (y - c)*(y - c)) < r) || (((x + c)*(x + c) + (y + c)*(y + c)) < r)));
        }
    }

    public class MySelfOrganizingNetwork : SelfOrganizingNetwork
    {
        public int aSize = 100;
        public Control control;
        private PatternsCollection patterns;

        public MySelfOrganizingNetwork(int aInputNodesCount, int aRowCount, int aColCount,
            double aInitialLearningRate, double aFinalLearningRate,
            int aInitialNeighborhoodSize, int aNeighborhoodReduceInterval,
            long aTrainingIterationsCount) : base(aInputNodesCount, aRowCount, aColCount,
                aInitialLearningRate, aFinalLearningRate,
                aInitialNeighborhoodSize, aNeighborhoodReduceInterval, aTrainingIterationsCount)
        {
        }

        public int p(double value)
        {
            return (int) Math.Round(value*aSize + aSize);
        }

        public void DrawPatterns(PatternsCollection pat, Graphics g)
        {
            foreach (var ptn in pat)
            {
                var pen = new Pen(Color.Black, 4);
                var pt = new Point(p(ptn.Input[0]), p(ptn.Input[1]));
                var pt2 = pt;
                pt2.Offset(2, 2);
                g.DrawLine(pen, pt, pt2);
                pen.Dispose();
            }
        }

        public override void Train(PatternsCollection patterns)
        {
            this.patterns = patterns;
            if (patterns != null)
                for (var i = 0; i < trainingIterations; i++)
                {
                    for (var j = 0; j < patterns.Count; j++)
                    {
                        SetValuesFromPattern(patterns[j]);
                        Run();
                        Learn();
                    }
                    Epoch(0);
                    if (PatternStyles.IsTerminated)
                        return;
                    if ((i%3) == 0)
                        ShowMeTheMovie();
                }
            ShowMeTheMovie();
        }

        public void ShowMeTheMovie()
        {
            if (PatternStyles.IsTerminated)
                return;
            if (control != null)
            {
                var g = control.CreateGraphics();
                g.Clear(control.BackColor);
                if (patterns != null)
                    DrawPatterns(patterns, g);
                for (var i = 0; i < rowsCount - 1; i++)
                    for (var j = 0; j < columsCount - 1; j++)
                    {
                        var pt1 = new Point(p(kohonenLayer[i, j].InLinks[0].Weight),
                            p(kohonenLayer[i, j].InLinks[1].Weight));
                        var pt2 = new Point(p(kohonenLayer[i + 1, j].InLinks[0].Weight),
                            p(kohonenLayer[i + 1, j].InLinks[1].Weight));
                        var pt3 = new Point(p(kohonenLayer[i + 1, j + 1].InLinks[0].Weight),
                            p(kohonenLayer[i + 1, j + 1].InLinks[1].Weight));
                        var pt4 = new Point(p(kohonenLayer[i, j + 1].InLinks[0].Weight),
                            p(kohonenLayer[i, j + 1].InLinks[1].Weight));
                        var pt5 = new Point(p(kohonenLayer[i, j].InLinks[0].Weight),
                            p(kohonenLayer[i, j].InLinks[1].Weight));
                        var gp = new GraphicsPath();
                        gp.StartFigure();
                        gp.AddLines(new[] {pt1, pt2, pt3, pt4, pt5});
                        gp.CloseFigure();
                        g.DrawPath(new Pen(Color.Brown), gp);
                        gp.Dispose();
                    }
                g.Flush();
                g.Dispose();
            }
            Application.DoEvents();
        }
    }

    public class Form1 : Form
    {
        public int InputsCount = 2;
        public int MapSize = 10;
        public int OutputsCount = 0;
        public int PatternsCount = 196;

        private int GetCheckedIndex()
        {
            foreach (Control c in groupBox1.Controls)
            {
                var r = (RadioButton) c;
                if ((r != null) && (r.Checked))
                    return r.TabIndex;
            }
            return 0;
        }

        private void SetPattern(PatternsCollection aPatterns, int i, double x, double y)
        {
            aPatterns[i - 1].Input[0] = x;
            aPatterns[i - 1].Input[1] = y;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            {
                var data = new PatternsCollection(PatternsCount, InputsCount, OutputsCount);
                var checkedItem = GetCheckedIndex();
                if (checkedItem > 6)
                    switch (checkedItem)
                    {
                        case 7:
                        {
                            PatternStyles.BuildEvenSquare(data);
                            break;
                        }
                        case 8:
                        {
                            PatternStyles.BuildEvenCircle(data);
                            break;
                        }
                    }
                else
                {
                    for (var i = 1; i <= PatternsCount; i++)
                    {
                        double x = 0;
                        double y = 0;
                        switch (checkedItem)
                        {
                            case 0:
                            {
                                PatternStyles.BuildSquare(out x, out y);
                                break;
                            }
                            case 1:
                            {
                                PatternStyles.BuildDimond(out x, out y);
                                break;
                            }
                            case 2:
                            {
                                PatternStyles.BuildPlus(out x, out y);
                                break;
                            }
                            case 3:
                            {
                                PatternStyles.BuildRing(out x, out y);
                                break;
                            }
                            case 4:
                            {
                                PatternStyles.BuildCircle(out x, out y);
                                break;
                            }
                            case 5:
                            {
                                PatternStyles.BuildTwoCircles(out x, out y);
                                break;
                            }
                            case 6:
                            {
                                PatternStyles.ChessBoard(out x, out y);
                                break;
                            }
                            default:
                                throw new ApplicationException("Please select pattern styly!");
                        }
                        SetPattern(data, i, x, y);
                    }
                }


                var LearningRateStart = 0.7;
                var LearningRateEnds = 0.05;
                var InitialNeighborhoodSize = 5;
                var NeighborhoodDecrRate = 5;
                var Iterations = 2000;
                var a = new MySelfOrganizingNetwork(InputsCount, MapSize, MapSize, LearningRateStart, LearningRateEnds,
                    InitialNeighborhoodSize, Iterations/NeighborhoodDecrRate, Iterations);
                a.control = panel1;
                a.Train(data);
                data = null;
                a = null;
            }
        }

        private void Form1_Closing(object sender, CancelEventArgs e)
        {
            PatternStyles.IsTerminated = true;
        }

        #region Hiden

        private Label label1;
        private Panel panel1;
        private GroupBox groupBox1;
        private RadioButton radioButton1;
        private RadioButton radioButton2;
        private RadioButton radioButton3;
        private RadioButton radioButton4;
        private RadioButton radioButton5;
        private RadioButton radioButton6;
        private RadioButton radioButton7;
        private RadioButton radioButton8;
        private RadioButton radioButton9;
        private Button button1;

        /// <summary>
        ///     Required designer variable.
        /// </summary>
        private readonly Container components = null;

        public Form1()
        {
            //
            // Required for Windows Form Designer support
            //
            InitializeComponent();

            //
            // TODO: Add any constructor code after InitializeComponent call
            //
        }

        /// <summary>
        ///     Clean up any resources being used.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (components != null)
                {
                    components.Dispose();
                }
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///     Required method for Designer support - do not modify
        ///     the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.label1 = new System.Windows.Forms.Label();
            this.panel1 = new System.Windows.Forms.Panel();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.radioButton9 = new System.Windows.Forms.RadioButton();
            this.radioButton8 = new System.Windows.Forms.RadioButton();
            this.radioButton7 = new System.Windows.Forms.RadioButton();
            this.radioButton6 = new System.Windows.Forms.RadioButton();
            this.radioButton5 = new System.Windows.Forms.RadioButton();
            this.radioButton4 = new System.Windows.Forms.RadioButton();
            this.radioButton3 = new System.Windows.Forms.RadioButton();
            this.radioButton2 = new System.Windows.Forms.RadioButton();
            this.radioButton1 = new System.Windows.Forms.RadioButton();
            this.button1 = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.label1.Dock = System.Windows.Forms.DockStyle.Top;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label1.ForeColor = System.Drawing.SystemColors.Window;
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(416, 32);
            this.label1.TabIndex = 1;
            this.label1.Text = "   SON training movie.";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // panel1
            // 
            this.panel1.Location = new System.Drawing.Point(8, 32);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(192, 192);
            this.panel1.TabIndex = 2;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.radioButton9,
                this.radioButton8,
                this.radioButton7,
                this.radioButton6,
                this.radioButton5,
                this.radioButton4,
                this.radioButton3,
                this.radioButton2,
                this.radioButton1
            });
            this.groupBox1.Location = new System.Drawing.Point(208, 40);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(200, 152);
            this.groupBox1.TabIndex = 3;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Pattern shape";
            // 
            // radioButton9
            // 
            this.radioButton9.Location = new System.Drawing.Point(104, 96);
            this.radioButton9.Name = "radioButton9";
            this.radioButton9.Size = new System.Drawing.Size(88, 24);
            this.radioButton9.TabIndex = 8;
            this.radioButton9.Text = "Even Circle";
            // 
            // radioButton8
            // 
            this.radioButton8.Location = new System.Drawing.Point(104, 72);
            this.radioButton8.Name = "radioButton8";
            this.radioButton8.Size = new System.Drawing.Size(88, 24);
            this.radioButton8.TabIndex = 7;
            this.radioButton8.Text = "Even Square";
            // 
            // radioButton7
            // 
            this.radioButton7.Location = new System.Drawing.Point(104, 48);
            this.radioButton7.Name = "radioButton7";
            this.radioButton7.Size = new System.Drawing.Size(88, 24);
            this.radioButton7.TabIndex = 6;
            this.radioButton7.Text = "Chess Board";
            // 
            // radioButton6
            // 
            this.radioButton6.Location = new System.Drawing.Point(104, 24);
            this.radioButton6.Name = "radioButton6";
            this.radioButton6.Size = new System.Drawing.Size(88, 24);
            this.radioButton6.TabIndex = 5;
            this.radioButton6.Text = "Two Circles";
            // 
            // radioButton5
            // 
            this.radioButton5.Location = new System.Drawing.Point(8, 120);
            this.radioButton5.Name = "radioButton5";
            this.radioButton5.TabIndex = 4;
            this.radioButton5.Text = "Circle";
            // 
            // radioButton4
            // 
            this.radioButton4.Location = new System.Drawing.Point(8, 96);
            this.radioButton4.Name = "radioButton4";
            this.radioButton4.TabIndex = 3;
            this.radioButton4.Text = "Ring";
            // 
            // radioButton3
            // 
            this.radioButton3.Location = new System.Drawing.Point(8, 72);
            this.radioButton3.Name = "radioButton3";
            this.radioButton3.TabIndex = 2;
            this.radioButton3.Text = "Plus Sign";
            // 
            // radioButton2
            // 
            this.radioButton2.Location = new System.Drawing.Point(8, 48);
            this.radioButton2.Name = "radioButton2";
            this.radioButton2.TabIndex = 1;
            this.radioButton2.Text = "Diamond";
            // 
            // radioButton1
            // 
            this.radioButton1.Checked = true;
            this.radioButton1.Location = new System.Drawing.Point(8, 24);
            this.radioButton1.Name = "radioButton1";
            this.radioButton1.TabIndex = 0;
            this.radioButton1.TabStop = true;
            this.radioButton1.Text = "Square";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(208, 200);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(120, 24);
            this.button1.TabIndex = 4;
            this.button1.Text = "Show me the movie";
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // Form1
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(416, 230);
            this.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.button1,
                this.groupBox1,
                this.panel1,
                this.label1
            });
            this.Name = "Form1";
            this.Text = "XPidea  -=Self-Organizing Network  Demo=-";
            this.Closing += new System.ComponentModel.CancelEventHandler(this.Form1_Closing);
            this.groupBox1.ResumeLayout(false);
            this.ResumeLayout(false);
        }

        #endregion

        /// <summary>
        ///     The main entry point for the application.
        /// </summary>
        [STAThread]
        private static void Main()
        {
            Application.Run(new Form1());
        }

        #endregion Hiden
    }
}