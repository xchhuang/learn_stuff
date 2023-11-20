extends RigidBody3D


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	var move_forward = Input.get_action_strength("ui_up");
	apply_central_force(move_forward * Vector3.FORWARD * 1200.0 * delta)
	
	var move_backward = Input.get_action_strength("ui_down");
	apply_central_force(move_backward * Vector3.BACK * 1200.0 * delta)
	
	var move_left = Input.get_action_strength("ui_left");
	apply_central_force(move_left * Vector3.LEFT * 1200.0 * delta)
	
	var move_right = Input.get_action_strength("ui_right");
	apply_central_force(move_right * Vector3.RIGHT * 1200.0 * delta)
	
