const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const WasmPackPlugin = require('@wasm-tool/wasm-pack-plugin');

module.exports = {
  entry: './www/index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'index.js',
  },
  mode: 'development',
  plugins: [
    new HtmlWebpackPlugin({
      template: './www/index.html'
    }),
    new WasmPackPlugin({
      crateDirectory: path.resolve(__dirname, '.'),
      outDir: path.resolve(__dirname, 'pkg'),
      extraArgs: '--target web'
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: './www/style.css', to: 'style.css' }
      ]
    })
  ],
  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    compress: true,
    port: 8080,
    open: true,
    hot: true
  },
  experiments: {
    asyncWebAssembly: true,
    syncWebAssembly: true
  }
};