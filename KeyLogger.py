import win32api as w32api
import time

keyList = ["\b"]

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
	keyList.append(char)

def keyCheck():
	keys = []
	for key in keyList:
		if w32api.GetAsyncKeyState(ord(key)):
			keys.append(key)
	return keys