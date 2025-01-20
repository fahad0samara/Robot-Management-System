class WarehouseSimulation {
    constructor() {
        this.canvas = document.getElementById('warehouseCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.socket = null;
        this.selectedRobot = null;
        this.showPaths = true;
        this.isPaused = false;
        this.speed = 1.0;
        this.cellWidth = 50;
        this.cellHeight = 50;
        this.colors = {
            empty: '#ffffff',
            storage: '#4a90e2',
            charging: '#f5a623',
            robot: '#2ecc71',
            selected: '#e74c3c',
            path: '#95a5a6'
        };
        
        this.setupCanvas();
        this.setupEventListeners();
        this.connectWebSocket();
    }
    
    setupCanvas() {
        // Make canvas responsive
        const resizeCanvas = () => {
            const container = this.canvas.parentElement;
            this.canvas.width = container.clientWidth;
            this.canvas.height = container.clientHeight;
        };
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    }
    
    setupEventListeners() {
        // Canvas click events
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.handleClick(x, y);
        });
        
        // Control buttons
        document.getElementById('pauseBtn').addEventListener('click', () => {
            this.isPaused = !this.isPaused;
            if (this.socket) this.socket.emit('pause', this.isPaused);
            document.getElementById('pauseBtn').textContent = this.isPaused ? 'Resume' : 'Pause';
        });
        
        document.getElementById('speedDown').addEventListener('click', () => {
            this.speed = Math.max(0.5, this.speed - 0.5);
            this.updateSpeedDisplay();
            if (this.socket) this.socket.emit('speed', this.speed);
        });
        
        document.getElementById('speedUp').addEventListener('click', () => {
            this.speed = Math.min(4.0, this.speed + 0.5);
            this.updateSpeedDisplay();
            if (this.socket) this.socket.emit('speed', this.speed);
        });
        
        document.getElementById('togglePaths').addEventListener('click', () => {
            this.showPaths = !this.showPaths;
            if (this.socket) this.socket.emit('toggle_paths', this.showPaths);
        });
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ':
                    document.getElementById('pauseBtn').click();
                    break;
                case 'ArrowUp':
                    document.getElementById('speedUp').click();
                    break;
                case 'ArrowDown':
                    document.getElementById('speedDown').click();
                    break;
                case 'p':
                case 'P':
                    document.getElementById('togglePaths').click();
                    break;
            }
        });
    }
    
    connectWebSocket() {
        this.socket = io('http://localhost:3000', {
            transports: ['websocket'],
            upgrade: false,
            reconnection: true,
            reconnectionAttempts: 5
        });

        this.socket.on('connect', () => {
            console.log('Connected to server');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
        });

        this.socket.on('update', (data) => {
            this.updateSimulation(data);
        });
    }
    
    updateSpeedDisplay() {
        document.getElementById('speedValue').textContent = `${this.speed.toFixed(1)}x`;
    }
    
    handleClick(x, y) {
        const gridX = Math.floor(x / this.cellWidth);
        const gridY = Math.floor(y / this.cellHeight);
        if (this.socket) this.socket.emit('click', { x: gridX, y: gridY });
    }
    
    updateSimulation(data) {
        if (!data || !data.warehouse || !data.robots) return;
        
        // Calculate cell dimensions based on canvas size and warehouse dimensions
        this.cellWidth = this.canvas.width / data.warehouse.dimensions[0];
        this.cellHeight = this.canvas.height / data.warehouse.dimensions[1];
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw warehouse grid
        this.drawWarehouse(data.warehouse);
        
        // Draw robots and paths
        this.drawRobots(data.robots);
        
        // Update status panel
        this.updateRobotStatus(data.robots);
    }
    
    drawWarehouse(warehouse) {
        const grid = warehouse.grid;
        for (let y = 0; y < grid.length; y++) {
            for (let x = 0; x < grid[y].length; x++) {
                const cell = grid[y][x];
                this.ctx.fillStyle = cell === 0 ? this.colors.empty :
                                   cell === 1 ? this.colors.storage :
                                   cell === 2 ? this.colors.charging : this.colors.empty;
                this.ctx.fillRect(x * this.cellWidth, y * this.cellHeight, this.cellWidth, this.cellHeight);
                this.ctx.strokeStyle = '#ddd';
                this.ctx.strokeRect(x * this.cellWidth, y * this.cellHeight, this.cellWidth, this.cellHeight);
            }
        }
    }
    
    drawRobots(robots) {
        robots.forEach(robot => {
            // Draw path if enabled
            if (this.showPaths && robot.path && robot.path.length > 0) {
                this.ctx.beginPath();
                this.ctx.moveTo(
                    (robot.position[0] + 0.5) * this.cellWidth,
                    (robot.position[1] + 0.5) * this.cellHeight
                );
                robot.path.forEach(point => {
                    this.ctx.lineTo(
                        (point[0] + 0.5) * this.cellWidth,
                        (point[1] + 0.5) * this.cellHeight
                    );
                });
                this.ctx.strokeStyle = this.colors.path;
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
            }
            
            // Draw robot
            const x = (robot.position[0] + 0.5) * this.cellWidth;
            const y = (robot.position[1] + 0.5) * this.cellHeight;
            const radius = Math.min(this.cellWidth, this.cellHeight) * 0.4;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, Math.PI * 2);
            this.ctx.fillStyle = robot.id === this.selectedRobot ? this.colors.selected : this.colors.robot;
            this.ctx.fill();
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
            
            // Draw robot ID
            this.ctx.fillStyle = '#fff';
            this.ctx.font = `${radius}px Arial`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(robot.id.toString(), x, y);
        });
    }
    
    updateRobotStatus(robots) {
        const statusDiv = document.getElementById('robotStatus');
        statusDiv.innerHTML = '';
        
        robots.forEach(robot => {
            const robotCard = document.createElement('div');
            robotCard.className = 'robot-card';
            robotCard.innerHTML = `
                <h3>Robot ${robot.id}</h3>
                <p>Status: ${robot.status}</p>
                <p>Battery: ${robot.battery_level.toFixed(1)}%</p>
            `;
            statusDiv.appendChild(robotCard);
        });
    }
}

// Initialize simulation when page loads
window.addEventListener('load', () => {
    const simulation = new WarehouseSimulation();
});
