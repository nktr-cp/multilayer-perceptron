const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const WasmPackPlugin = require('@wasm-tool/wasm-pack-plugin');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  const isPages = process.env.BUILD_TARGET === 'pages';
  
  return {
    entry: './www/index.js',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction ? '[name].[contenthash].js' : 'index.js',
      clean: true,
      // Use relative paths for GitHub Pages
      publicPath: isPages ? './' : '/',
    },
    mode: argv.mode || 'development',
    optimization: {
      splitChunks: isProduction ? {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
          wasm: {
            test: /\.wasm$/,
            name: 'wasm',
            chunks: 'all',
          }
        },
      } : false,
      minimize: isProduction,
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: './www/index.html',
        minify: isProduction ? {
          removeComments: true,
          collapseWhitespace: true,
          removeRedundantAttributes: true,
          useShortDoctype: true,
          removeEmptyAttributes: true,
          removeStyleLinkTypeAttributes: true,
          keepClosingSlash: true,
          minifyJS: true,
          minifyCSS: true,
          minifyURLs: true,
        } : false,
        inject: 'head',
        scriptLoading: 'defer'
      }),
      new WasmPackPlugin({
        crateDirectory: path.resolve(__dirname, '.'),
        outDir: path.resolve(__dirname, 'pkg'),
        extraArgs: isProduction ? '--target web --release' : '--target web',
        forceMode: isProduction ? 'production' : 'development',
        env: {
          RUSTFLAGS: '--cfg=web_sys_unstable_apis'
        }
      }),
      new CopyWebpackPlugin({
        patterns: [
          { from: './www/style.css', to: 'style.css' },
          // Add any other static assets here
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
    },
    performance: {
      hints: isProduction ? 'warning' : false,
      maxAssetSize: 500000,
      maxEntrypointSize: 500000,
    },
    resolve: {
      extensions: ['.js', '.wasm'],
    }
  };
};