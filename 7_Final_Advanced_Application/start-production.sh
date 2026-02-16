#!/bin/bash

# Fossil AI Hub - Deployment Script with Access Information
# This script starts the services and displays access information

set -e

echo "🚀 Starting Fossil AI Hub Production Deployment..."
echo "=================================================="

# Start services in detached mode
docker compose -f docker-compose.production.yml up -d

# Wait a moment for services to initialize
sleep 3

# Get the local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || ipconfig | grep -A1 "Ethernet adapter" | grep "IPv4" | awk '{print $NF}' 2>/dev/null || echo "localhost")

echo ""
echo "✅ Services started successfully!"
echo "=================================="
echo ""
echo "📱 Fossil AI Hub Access URLs:"
echo "   Frontend (Web UI):"
echo "     • Local:    http://localhost:8080"
echo "     • Network:  http://$LOCAL_IP:8080"
echo ""
echo "🔧 Backend API:"
echo "     • Local:    http://localhost:5000"
echo "     • Network:  http://$LOCAL_IP:5000"
echo ""
echo "📊 Health Check:"
echo "     • http://localhost:5000/api/health"
echo ""
echo "📋 Management Commands:"
echo "     • View logs:  docker compose -f docker-compose.production.yml logs -f"
echo "     • Stop:       docker compose -f docker-compose.production.yml down"
echo "     • Update:     docker compose -f docker-compose.production.yml pull && docker compose -f docker-compose.production.yml up -d"
echo ""
echo "⏳ Waiting for services to be ready..."

# Wait for backend health check
echo -n "   Backend: "
for i in {1..30}; do
    if curl -sf http://localhost:5000/api/health >/dev/null 2>&1; then
        echo "✅ Ready!"
        break
    fi
    echo -n "."
    sleep 2
done

# Check if frontend is responding
echo -n "   Frontend: "
for i in {1..10}; do
    if curl -sf http://localhost:8080 >/dev/null 2>&1; then
        echo "✅ Ready!"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "🎉 Fossil AI Hub is now running!"
echo "   Open your browser and go to: http://$LOCAL_IP:8080"
echo "=================================="