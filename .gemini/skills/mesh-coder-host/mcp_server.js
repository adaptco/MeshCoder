const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * MCP Server for the MeshCoder Host.
 * Acts as a bridge to the TS Gateway and manages backend Python services.
 */

const rootDir = path.join(__dirname, '../../..');
const gatewayDir = path.join(rootDir, 'mcp/gateway');
const servicesDir = path.join(rootDir, 'mcp/services');

// 1. Launch Python Services in background
/**
 * @param {string} name
 * @param {string} dir
 * @param {string} script
 */
function launchService(name, dir, script) {
    console.error(`[Host] Starting ${name} service...`);
    const serviceProcess = spawn('python3', [script], {
        cwd: dir,
        stdio: 'inherit', // Let logs flow to stderr
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });
    
    serviceProcess.on('error', (err) => {
        console.error(`[Host] Failed to start ${name}:`, err);
    });

    return serviceProcess;
}

const searchService = launchService('Search', path.join(servicesDir, 'search-mcp'), 'main.py');
const parquetService = launchService('Parquet', path.join(servicesDir, 'parquet-service'), 'main.py');
const gmailService = launchService('Gmail', path.join(servicesDir, 'gmail-mcp'), 'main.py');

// 2. Launch TS Gateway on stdio
// We proxy our stdio to the TS Gateway's stdio
console.error('[Host] Starting TS Gateway bridge...');

const gatewayProcess = spawn('npx', ['ts-node', 'src/index.ts'], {
    cwd: gatewayDir,
    stdio: ['inherit', 'inherit', 'inherit'], // Bridge all streams
    env: { ...process.env }
});

gatewayProcess.on('exit', (code) => {
    console.error(`[Host] Gateway exited with code ${code}`);
    searchService.kill();
    parquetService.kill();
    gmailService.kill();
    process.exit(code);
});

// Handle termination
process.on('SIGINT', () => {
    searchService.kill();
    parquetService.kill();
    gmailService.kill();
    gatewayProcess.kill();
    process.exit(0);
});
