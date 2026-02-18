module.exports = {
  apps: [{
    name: 'voter-extraction-pipeline',
    script: 'venv/bin/python',
    args: 'voter_extraction_pipeline.py',
    interpreter: 'none',
    cwd: '/home/ec2-user/Voter_extract',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '4G',
    env: {
      NODE_ENV: 'production'
    },
    error_file: './logs/pm2-error.log',
    out_file: './logs/pm2-out.log',
    log_file: './logs/pm2-combined.log',
    time: true,
    merge_logs: true,

    // Restart behavior
    restart_delay: 5000,
    max_restarts: 10,
    min_uptime: 60000,

    // Auto-restart on specific errors
    kill_timeout: 5000,
    listen_timeout: 10000,
  }]
};
