#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>

#include <linux/input-event-codes.h>

#ifndef KEYBOARD_H_
#define KEYBOARD_H_

class Keyboard
{
public:

	// Create keyboard device.
	static Keyboard* Create(const char* path="/dev/input/by-path/platform-i8042-serio-0-event-kbd");

	// Constructor.
	Keyboard();

	// Destructor.
	~Keyboard();

	// Poll the keyboard for updates.
	bool Poll(uint32_t timeout = 0);

	// Check if a key is pressed.
	bool KeyDown(uint32_t code) const;

protected:

	static const int MAX_KEYS = 256;

	int key_map_[MAX_KEYS];
	int found_device_;

	std::string path_;

};

#endif
