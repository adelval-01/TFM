import asyncio

from livekit import api 
from livekit.protocol.sip import CreateSIPInboundTrunkRequest, SIPInboundTrunkInfo, CreateSIPDispatchRuleRequest, SIPDispatchRule, SIPDispatchRuleDirect, ListSIPInboundTrunkRequest, DeleteSIPTrunkRequest


async def main():
  livekit_api = api.LiveKitAPI(
    url="wss://test-tfm-3lii83j0.livekit.cloud",
    api_key="APIaRjG6ptSKLTy",
    api_secret="VAovqneYo91unfwHFCGmlJgevds1CZDUIbeufPUuThoE"
  )

  trunk_id = '<trunk_id>'
  room_name = 'TFM'


  # Create a dispatch rule to place all callers in the same room
  rule = SIPDispatchRule(
    dispatch_rule_direct = SIPDispatchRuleDirect(
      room_name = 'TFM',
      # pin = ''
    )   
  )

  request = CreateSIPDispatchRuleRequest(
    rule = rule,
    name = 'My dispatch rule',
    # trunk_ids = [ 
    #   trunk_id,
    # ],  
    hide_phone_number = False
  )

  try:
    dispatchRule = await livekit_api.sip.create_sip_dispatch_rule(request)
    print(f"Successfully created {dispatchRule}")
  except api.twirp_client.TwirpError as e:
    print(f"{e.code} error: {e.message}")

  trunk = SIPInboundTrunkInfo(
    name = "Inbound trunk",
    numbers = ['+12025688661'],
  )
  
  request = CreateSIPInboundTrunkRequest(
    trunk = trunk
  )
  print(f"Created SIP inbound trunk {trunk}")

  trunk = await livekit_api.sip.create_sip_inbound_trunk(request)

  # request = DeleteSIPTrunkRequest(
  #   sip_trunk_id = "ST_UUSDbPr48NuD"
  # )
  # print(f"Removed SIP inbound trunk {trunk}")

  # trunk = await livekit_api.sip.delete_sip_trunk(request)


  rules = await livekit_api.sip.list_sip_inbound_trunk(
    ListSIPInboundTrunkRequest()
  )
  print(f"{rules}")
  
  await livekit_api.aclose()

asyncio.run(main())