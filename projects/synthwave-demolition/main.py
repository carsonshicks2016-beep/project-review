import sys
import random
import math
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import Vec3, Point3, Vec4, TransformState, NodePath, Quat, LColor
from panda3d.core import DirectionalLight, AmbientLight, PointLight
from panda3d.core import TextNode, loadPrcFileData
from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape, BulletVehicle, ZUp, BulletGenericConstraint
from panda3d.bullet import BulletDebugNode
import simplepbr

# Configure Panda3D for better visuals
loadPrcFileData('', 'win-size 1280 720')
loadPrcFileData('', 'framebuffer-multisample 1')
loadPrcFileData('', 'multisamples 4')

class PhysicsApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Initialize PBR for synthwave look
        self.pipeline = simplepbr.init(
            use_normal_maps=True,
            enable_shadows=True,
            use_emission_maps=True,
            msaa_samples=4
        )
        self.setBackgroundColor(0.02, 0.0, 0.05, 1) # Dark purple synthwave sky
        
        # Physics setup
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        
        # Visual setup
        self.setup_lighting()
        
        # Game state
        self.cars = []
        self.breakable_parts = []
        self.particles = []
        
        self.player_health = 100
        self.player_score = 0
        
        # Player
        self.player_car = self.create_car(Point3(0, 0, 2), is_player=True)
        self.steering = 0.0
        self.engine_force = 0.0
        
        # Environment
        self.setup_environment()
        
        # UI
        self.setup_ui()
        
        # Input
        self.accept('escape', sys.exit)
        self.accept('d', self.set_steering, [-1])
        self.accept('a', self.set_steering, [1])
        self.accept('d-up', self.set_steering, [0])
        self.accept('a-up', self.set_steering, [0])
        self.accept('w', self.set_engine, [1])
        self.accept('s', self.set_engine, [-1])
        self.accept('w-up', self.set_engine, [0])
        self.accept('s-up', self.set_engine, [0])
        self.accept('space', self.jump) # Fun stunt mechanic
        
        # Enemies
        for i in range(5):
            pos = Point3(random.uniform(-40, 40), random.uniform(20, 80), 2)
            self.create_car(pos, is_player=False)
            
        # Camera
        self.cam.setPos(0, -20, 10)
        self.cam_shake = 0.0
        
        # Update Task
        self.taskMgr.add(self.update, 'update')
        
        # Preload the box model
        self.box_model = self.loader.loadModel("box")

    def setup_lighting(self):
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.1, 0.4, 1)) # Synthwave ambient
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Main sun/moon
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.6, 0.9, 1))
        dlight.getLens().setNearFar(1, 200)
        dlight.getLens().setFilmSize(100, 100)
        dlight.setShadowCaster(True, 1024, 1024)
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -45, -45)
        self.render.setLight(dlnp)
        
    def setup_ui(self):
        self.speed_text = OnscreenText(text='0 MPH', pos=(1.0, -0.9), scale=0.1, fg=(0, 1, 1, 1), align=TextNode.ARight, mayChange=True)
        self.health_text = OnscreenText(text='HEALTH: 100%', pos=(-1.2, -0.9), scale=0.08, fg=(0, 1, 0.5, 1), align=TextNode.ALeft, mayChange=True)
        self.score_text = OnscreenText(text='SCORE: 0', pos=(-1.2, 0.9), scale=0.08, fg=(1, 0.2, 0.8, 1), align=TextNode.ALeft, mayChange=True)
        self.game_over_text = OnscreenText(text='', pos=(0, 0), scale=0.2, fg=(1, 0, 0, 1), align=TextNode.ACenter, mayChange=True)

    def create_visual_box(self, parent, scale, color):
        if not hasattr(self, 'box_model'):
            self.box_model = self.loader.loadModel("box")
        
        visual = self.box_model.copyTo(parent)
        
        # Built-in box is usually from -1 to 1 (size 2), so if BulletBoxShape is half-extents, we scale by Bullet half-extents.
        # Wait, BulletBoxShape takes half-extents. So a (1, 2, 0.5) BulletBox is actually 2x4x1 in total size.
        # The built-in Panda box is 2x2x2 (from -1 to 1). So scaling by Bullet half-extents (1, 2, 0.5) makes the box 2x4x1! Perfect!
        visual.setScale(scale)
        visual.setColor(color)
        return visual

    def setup_environment(self):
        # Ground (Grid)
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        node = BulletRigidBodyNode('Ground')
        node.addShape(shape)
        np = self.render.attachNewNode(node)
        self.world.attachRigidBody(node)
        
        # Ground visual
        ground_vis = self.create_visual_box(np, Vec3(1000, 1000, 0.1), LColor(0.05, 0.05, 0.05, 1.0))
        ground_vis.setPos(0, 0, -0.1)
        
        # City Grid
        for x in range(-10, 10):
            for y in range(-10, 10):
                if x == 0 and y == 0: continue # keep spawn clear
                
                # Buildings
                if random.random() < 0.3:
                    h = random.uniform(5, 20)
                    w = random.uniform(2, 6)
                    d = random.uniform(2, 6)
                    
                    box_shape = BulletBoxShape(Vec3(w, d, h))
                    box_node = BulletRigidBodyNode('Building')
                    box_node.addShape(box_shape)
                    box_np = self.render.attachNewNode(box_node)
                    box_np.setPos(x * 20, y * 20, h)
                    self.world.attachRigidBody(box_node)
                    
                    # Neon color for building
                    color = LColor(random.uniform(0.1, 0.5), random.uniform(0.1, 0.8), random.uniform(0.5, 1.0), 1)
                    if random.random() < 0.2: color = LColor(1.0, 0.2, 0.6, 1) # Hot pink
                    
                    self.create_visual_box(box_np, Vec3(w, d, h), color)
                
                # Ramps
                elif random.random() < 0.1:
                    ramp_shape = BulletBoxShape(Vec3(4, 8, 0.5))
                    ramp_node = BulletRigidBodyNode('Ramp')
                    ramp_node.addShape(ramp_shape)
                    ramp_np = self.render.attachNewNode(ramp_node)
                    ramp_np.setPos(x * 20, y * 20, 0.5)
                    ramp_np.setHpr(0, 20, 0) # Tilt
                    self.world.attachRigidBody(ramp_node)
                    self.create_visual_box(ramp_np, Vec3(4, 8, 0.5), LColor(1.0, 0.8, 0.2, 1))
                    
    def create_car(self, pos, is_player=False):
        # Chassis
        chassis_half = Vec3(1.0, 2.0, 0.5)
        chassis_shape = BulletBoxShape(chassis_half)
        chassis_node = BulletRigidBodyNode('Player' if is_player else 'Enemy')
        chassis_node.addShape(chassis_shape)
        chassis_node.setMass(1000.0)
        chassis_node.setDeactivationEnabled(False)
        chassis_np = self.render.attachNewNode(chassis_node)
        chassis_np.setPos(pos)
        self.world.attachRigidBody(chassis_node)
        
        # Chassis Visual
        chassis_color = LColor(0.1, 0.8, 1.0, 1) if is_player else LColor(1.0, 0.2, 0.2, 1)
        self.create_visual_box(chassis_np, chassis_half, chassis_color)
        
        # Vehicle
        vehicle = BulletVehicle(self.world, chassis_node)
        vehicle.setCoordinateSystem(ZUp)
        self.world.attachVehicle(vehicle)
        
        # Wheels
        w_radius = 0.5
        front_left = Point3(-1.2, 1.4, -0.3)
        front_right = Point3(1.2, 1.4, -0.3)
        rear_left = Point3(-1.2, -1.4, -0.3)
        rear_right = Point3(1.2, -1.4, -0.3)
        
        w_fl = self.add_wheel(vehicle, front_left, True, w_radius)
        w_fr = self.add_wheel(vehicle, front_right, True, w_radius)
        w_rl = self.add_wheel(vehicle, rear_left, False, w_radius)
        w_rr = self.add_wheel(vehicle, rear_right, False, w_radius)

        car_data = {
            'chassis_np': chassis_np,
            'chassis_node': chassis_node,
            'vehicle': vehicle,
            'is_player': is_player,
            'steering': 0.0,
            'engine_force': 0.0,
            'top_speed_mult': 1.0,
            'dead': False,
            'wheel_visuals': [w_fl, w_fr, w_rl, w_rr]
        }
        
        self.cars.append(car_data)
        
        # Add Breakable parts
        part_color = LColor(0.2, 0.2, 0.2, 1)
        self.add_breakable_part(car_data, Point3(0, 2.3, 0), Vec3(1.0, 0.3, 0.4), "FrontBumper", part_color)
        self.add_breakable_part(car_data, Point3(0, -2.3, 0), Vec3(1.0, 0.3, 0.4), "RearBumper", part_color)
        self.add_breakable_part(car_data, Point3(-1.3, 0, 0), Vec3(0.3, 1.5, 0.4), "LeftDoor", chassis_color)
        self.add_breakable_part(car_data, Point3(1.3, 0, 0), Vec3(0.3, 1.5, 0.4), "RightDoor", chassis_color)
        self.add_breakable_part(car_data, Point3(0, 0, 0.8), Vec3(0.9, 1.0, 0.3), "Roof", chassis_color)
        
        return car_data

    def add_wheel(self, vehicle, pos, is_front, radius):
        wheel = vehicle.createWheel()
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(is_front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(40.0)
        wheel.setSuspensionStiffness(50.0)
        wheel.setWheelsDampingRelaxation(2.3)
        wheel.setWheelsDampingCompression(4.4)
        wheel.setFrictionSlip(3.0) # slightly drifty
        wheel.setRollInfluence(0.1)
        
        # Wheel visual
        # Panda3D Bullet vehicle wheels don't automatically sync a visual node natively without extra setup.
        # We can attach a node to the chassis and update its pos/hpr every frame, or rely on BulletVehicle's getWheelTransform
        vis_np = self.render.attachNewNode("wheel_vis")
        self.create_visual_box(vis_np, Vec3(0.3, radius, radius), LColor(0.1, 0.1, 0.1, 1))
        
        idx = vehicle.getNumWheels() - 1
        return {'wheel': wheel, 'np': vis_np, 'idx': idx}

    def add_breakable_part(self, car_data, pos, size, name, color):
        chassis_np = car_data['chassis_np']
        chassis_node = car_data['chassis_node']
        
        part_shape = BulletBoxShape(size)
        part_node = BulletRigidBodyNode(f"{chassis_node.getName()}_{name}")
        part_node.setMass(20.0)
        part_node.addShape(part_shape)
        part_np = self.render.attachNewNode(part_node)
        
        part_np.setPos(chassis_np, pos)
        part_np.setQuat(chassis_np.getQuat())
        
        self.world.attachRigidBody(part_node)
        self.create_visual_box(part_np, size, color)
        
        frameA = TransformState.makePos(pos)
        frameB = TransformState.makePos(Point3(0,0,0))
        
        constraint = BulletGenericConstraint(chassis_node, part_node, frameA, frameB, True)
        
        # Lock all axes
        for i in range(3):
            constraint.setLinearLimit(i, 0, 0)
            constraint.setAngularLimit(i, 0, 0)
            
        self.world.attachConstraint(constraint)
        
        self.breakable_parts.append({
            'node': part_node,
            'np': part_np,
            'constraint': constraint,
            'car_data': car_data,
            'broken': False,
            'name': name
        })

    def spawn_particles(self, pos, color, count=10):
        for _ in range(count):
            p_np = self.render.attachNewNode("particle")
            p_size = random.uniform(0.1, 0.3)
            self.create_visual_box(p_np, Vec3(p_size, p_size, p_size), color)
            p_np.setPos(pos + Vec3(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)))
            
            vel = Vec3(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(5, 15))
            self.particles.append({'np': p_np, 'vel': vel, 'life': 1.0})

    def set_steering(self, val):
        self.steering = val * 35.0 

    def set_engine(self, val):
        self.engine_force = val * 3000.0

    def jump(self):
        # Apply upwards impulse if on ground
        if self.player_health > 0:
            self.player_car['chassis_node'].applyCentralImpulse(Vec3(0, 0, 5000))
            self.spawn_particles(self.player_car['chassis_np'].getPos(), LColor(1, 1, 0, 1), 20)

    def update(self, task):
        dt = globalClock.getDt()
        
        if self.player_health > 0:
            # Player controls
            eff_force = self.engine_force * self.player_car['top_speed_mult']
            self.player_car['vehicle'].setSteeringValue(self.steering, 0)
            self.player_car['vehicle'].setSteeringValue(self.steering, 1)
            self.player_car['vehicle'].applyEngineForce(eff_force, 2)
            self.player_car['vehicle'].applyEngineForce(eff_force, 3)
            
            # Speedometer UI
            speed = self.player_car['vehicle'].getCurrentSpeedKmHour()
            self.speed_text.setText(f"{int(abs(speed))} MPH")
        else:
            self.player_car['vehicle'].applyEngineForce(0, 2)
            self.player_car['vehicle'].applyEngineForce(0, 3)
            self.game_over_text.setText("GAME OVER\nCAR DESTROYED")
        
        # Update Wheel visuals
        for car in self.cars:
            for wv in car['wheel_visuals']:
                # Sync wheel visual to physics wheel
                ts = car['vehicle'].getWheel(wv['idx']).getWorldTransform()
                wv['np'].setMat(ts)
        
        # Update Enemies (AI)
        player_pos = self.player_car['chassis_np'].getPos()
        for car in self.cars:
            if car['is_player'] or car['dead']: continue
            
            car_pos = car['chassis_np'].getPos()
            to_player = player_pos - car_pos
            
            # Very basic avoidance/pathing: if blocked, reverse
            speed = car['vehicle'].getCurrentSpeedKmHour()
            
            dist = to_player.length()
            if dist > 5.0 and dist < 150.0:
                forward = car['chassis_np'].getQuat().getForward()
                to_player.normalize()
                
                cross = forward.cross(to_player)
                steer_val = 0
                if cross.z > 0.1: steer_val = 35.0
                elif cross.z < -0.1: steer_val = -35.0
                
                # If stuck, reverse
                force = 2000.0 * car['top_speed_mult']
                if abs(speed) < 2 and random.random() < 0.05:
                    force = -2000.0
                    steer_val *= -1
                    
                car['vehicle'].setSteeringValue(steer_val, 0)
                car['vehicle'].setSteeringValue(steer_val, 1)
                car['vehicle'].applyEngineForce(force, 2)
                car['vehicle'].applyEngineForce(force, 3)
            else:
                car['vehicle'].applyEngineForce(0, 2)
                car['vehicle'].applyEngineForce(0, 3)
                
            # If enemy falls off world, kill them
            if car_pos.z < -20:
                car['dead'] = True
        
        # Breakable parts logic
        for part in self.breakable_parts:
            if part['broken']: continue
            
            result = self.world.contactTest(part['node'])
            if result.getNumContacts() > 0:
                # Calculate rough impact force
                impact = 0
                for contact in result.getContacts():
                    node0 = contact.getNode0()
                    node1 = contact.getNode1()
                    # Ignore ground drag for breaking
                    if node0.getName() == 'Ground' or node1.getName() == 'Ground':
                        continue
                    impact += 1
                
                if impact > 0: # Any solid hit with building/other car
                    part['broken'] = True
                    self.world.removeConstraint(part['constraint'])
                    self.spawn_particles(part['np'].getPos(), LColor(1, 0.5, 0, 1), 15)
                    self.cam_shake = 0.5
                    
                    # Meaningful Damage Logic
                    car = part['car_data']
                    if car['is_player']:
                        self.player_health -= 20
                        if part['name'] == "FrontBumper":
                            car['top_speed_mult'] = 0.7
                            
                        # Update UI
                        h_col = (0, 1, 0.5, 1) if self.player_health > 50 else (1, 0, 0, 1)
                        self.health_text.setText(f"HEALTH: {max(0, self.player_health)}%")
                        self.health_text.setFg(h_col)
                        
                        if self.player_health <= 0:
                            self.spawn_particles(car['chassis_np'].getPos(), LColor(1, 0, 0, 1), 50)
                            self.cam_shake = 1.0
                    else:
                        # Enemy takes damage, maybe dies
                        if random.random() < 0.3:
                            car['dead'] = True
                            self.player_score += 100
                            self.score_text.setText(f"SCORE: {self.player_score}")
                            self.spawn_particles(car['chassis_np'].getPos(), LColor(1, 0.5, 0, 1), 30)

        # Update Particles
        for p in self.particles[:]:
            p['life'] -= dt
            if p['life'] <= 0:
                p['np'].removeNode()
                self.particles.remove(p)
            else:
                p['vel'].z -= 9.81 * dt # gravity
                p['np'].setPos(p['np'].getPos() + p['vel'] * dt)
                p['np'].setScale(p['life']) # shrink
        
        # Step Physics
        self.world.doPhysics(dt, 10, 1.0/120.0)
        
        # Camera Follow Player with Shake
        p_pos = self.player_car['chassis_np'].getPos()
        p_quat = self.player_car['chassis_np'].getQuat()
        forward = p_quat.getForward()
        
        target_cam_pos = p_pos - forward * 25.0 + Vec3(0, 0, 8.0)
        
        # Shake
        if self.cam_shake > 0:
            target_cam_pos += Vec3(random.uniform(-self.cam_shake, self.cam_shake), 
                                   random.uniform(-self.cam_shake, self.cam_shake), 
                                   random.uniform(-self.cam_shake, self.cam_shake))
            self.cam_shake -= dt * 2.0
            if self.cam_shake < 0: self.cam_shake = 0
            
        current_cam_pos = self.cam.getPos()
        self.cam.setPos(current_cam_pos + (target_cam_pos - current_cam_pos) * dt * 5.0)
        self.cam.lookAt(p_pos + Vec3(0, 0, 2.0))
        
        return task.cont

if __name__ == '__main__':
    app = PhysicsApp()
    app.run()
