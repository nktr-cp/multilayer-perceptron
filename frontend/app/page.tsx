"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { NetworkVisualizer } from "@/components/network-visualizer"
import { LayerControlPanel } from "@/components/layer-control-panel"
import { TrainingDashboard } from "@/components/training-dashboard"
import { DatasetSelector } from "@/components/dataset-selector"
import { OptimizerPanel } from "@/components/optimizer-panel"
import { TaskSwitcher } from "@/components/task-switcher"
import { ConsoleOutput } from "@/components/console-output"
import { NeuralBackground } from "@/components/neural-background"
import { DatasetGenerator } from "@/components/dataset-generator"
import { Brain, Github, BookOpen } from "lucide-react"

export default function MLPDemo() {
  const [layers, setLayers] = useState([4, 8, 8, 3])
  const [activationFn, setActivationFn] = useState("relu")
  const [taskType, setTaskType] = useState<"binary" | "multiclass" | "regression">("multiclass")
  const [isTraining, setIsTraining] = useState(false)
  const [dataset, setDataset] = useState("iris")
  const [optimizer, setOptimizer] = useState("adam")
  const [learningRate, setLearningRate] = useState(0.01)

  return (
    <div className="relative min-h-screen bg-background neural-grid overflow-hidden">
      <NeuralBackground />

      <div className="relative z-10">
        {/* Hero Section */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative border-b border-border/50 backdrop-blur-sm"
        >
          <div className="container mx-auto px-4 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Brain className="w-8 h-8 text-primary animate-neural-pulse" />
                  <div className="absolute inset-0 blur-xl bg-primary/30 animate-pulse-glow" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-foreground">Neural Lab</h1>
                  <p className="text-sm text-muted-foreground">Train Intelligence in Your Browser</p>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  <BookOpen className="w-5 h-5" />
                </a>
                <a href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                  <Github className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>
        </motion.header>

        <div className="container mx-auto px-4 py-8">
          <div className="space-y-6 max-w-7xl mx-auto">
            {/* Data Generation */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">01</span> Data Generation
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <TaskSwitcher value={taskType} onChange={setTaskType} />
                  <DatasetSelector value={dataset} onChange={setDataset} taskType={taskType} />
                </div>
                <DatasetGenerator />
              </div>
            </motion.div>

            {/* Layer Controller / Optimizer */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">02</span> Layer Controller / Optimizer
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <LayerControlPanel
                  layers={layers}
                  onLayersChange={setLayers}
                  activationFn={activationFn}
                  onActivationChange={setActivationFn}
                />
                <OptimizerPanel
                  optimizer={optimizer}
                  onOptimizerChange={setOptimizer}
                  learningRate={learningRate}
                  onLearningRateChange={setLearningRate}
                />
              </div>
            </motion.div>

            {/* Network Architecture */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">03</span> Network Architecture
              </h2>
              <NetworkVisualizer layers={layers} isTraining={isTraining} activationFn={activationFn} />
            </motion.div>

            {/* Training Metrics */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">04</span> Training Metrics
              </h2>
              <TrainingDashboard isTraining={isTraining} onToggleTraining={() => setIsTraining(!isTraining)} />
            </motion.div>

            {/* Console */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">05</span> Console
              </h2>
              <ConsoleOutput isTraining={isTraining} />
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}
